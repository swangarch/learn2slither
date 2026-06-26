#!/usr/bin/python3

import os
import json
from datetime import datetime

import numpy as np
from numpy import ndarray as array
import matplotlib.pyplot as plt

# activation_func.py
from .activation_func import (
    relu,
    relu_deriv,
    sigmoid,
    sigmoid_deriv,
    leaky_relu,
    leaky_relu_deriv,
    softmax,
    activ_deriv,
)

# nnUtils.py
from .nnUtils import (
    network,
    create_bias,
    forward_layer,
    gradient_descent,
    split_dataset,
    shuffle_data,
    accuracy_1d,
    loss,
    ce_loss,
    bce_loss,
    mse_loss,
    cmp_correct,
)


class NN:
    """Neural network class, which can perform training and prediction
    for both classification and regression tasks."""

    deriv_map = {
            relu: relu_deriv,
            sigmoid: sigmoid_deriv,
            leaky_relu: leaky_relu_deriv
        }

    def __init__(self, shape: list, activation_functions: list,
                 init_methods: list, classification: bool = False,
                 loss: str = "MeanSquareError"):
        """Init a multilayer perceptron."""

        if len(shape) < 4:
            raise ValueError("Net shape too short.")
        if len(shape) != len(activation_functions) + 1:
            raise ValueError("Mismatched net shape and activation functions.")

        self.net_shape = shape
        self.init_methods = init_methods
        print("[NET] ", self.net_shape)
        self.activ_funcs = activation_functions

        self.nets = network(self.net_shape, self.init_methods)
        self.len_nets = len(self.nets)
        self.len_out = self.net_shape[-1]
        self.biases = create_bias(self.net_shape, 0)
        self.loss_func = loss

        self.graph_loss_train = []
        self.graph_loss_test = []
        self.graph_acc_train = []
        self.graph_acc_test = []
        self.graph_epoch = []

        self.loss_threshold = 0.000001
        self.loss_test = None
        self.loss_train = None
        self.plt = plt

        self.classification = classification

    def check_train_params(self, inputs: array, truths: array) -> None:
        """Check training parameters."""

        if len(inputs) != len(truths):
            raise ValueError("Mismatched training dataset")
        if self.net_shape[-1] != 1 and self.loss_func == "BinaryCrossEntropy":
            info = "BinaryCrossEntropy only adapt 1d for training output."
            raise ValueError(info)
        if (
            self.activ_funcs[-1] == "softmax" and
            self.loss_func != "CategoricalCrossEntropy"
        ):
            info = "Softmax only adapte to CategoricalCrossEntropy loss."
            raise ValueError(info)

    def train_batch(self, inputs: array, truths: array,
                    learning_rate: float = 0.01) -> None:
        """Train a batch, the inputs and truths have to be already chunked
        into batch. This function will perform feed foward, back probagation,
        and gradient descent, the process to train the model."""

        inputs_b = inputs.T
        # (batch_size, features) -> (features, batch_size)
        t_b = truths.T
        # -----------------------------forward ----------------------------
        act = [inputs_b]
        Bgrads = []
        Wgrads = []
        for i in range(self.len_nets):
            act.append(forward_layer(self.nets[i],
                                     act[i],
                                     self.biases[i],
                                     self.activ_funcs[i]))
        # -----------------------------forward end-------------------------
        # -----------------------------back probab ------------------------
        # Last layer
        cce = "CategoricalCrossEntropy"
        bce = "BinaryCrossEntropy"
        if self.loss_func == cce and self.activ_funcs[-1] == softmax:
            local_grad = act[-1] - t_b
        elif self.loss_func == bce and self.activ_funcs[-1] == sigmoid:
            local_grad = act[-1] - t_b
        else:
            local_grad = (act[-1] - t_b) * activ_deriv(self.activ_funcs[-1],
                                                       act[-1], NN.deriv_map)
        # last layer difference
        Bgrads.append(np.mean(local_grad, axis=1, keepdims=True))
        Wgrads.append(local_grad @ act[-2].T / len(inputs))
        for i in range(self.len_nets - 1, 0, -1):
            grad_prev_layer = self.nets[i].T @ local_grad
            # cal the loss of prev layer
            local_grad = grad_prev_layer * activ_deriv(self.activ_funcs[i - 1],
                                                       act[i], NN.deriv_map)
            Bgrads.append(np.mean(local_grad, axis=1, keepdims=True))
            Wgrads.append(local_grad @ act[i - 1].T / len(inputs))
        # -----------------------------back probab end----------------------
        gradient_descent(self.nets, self.biases, Wgrads[::-1],
                         Bgrads[::-1], learning_rate)

    def inference(self, inputs: array) -> array:
        """After training, use weights to do inference. The result will be raw
        from neural network. If onehot encoding and softmax is applied, the
        category will need post argmax."""

        activ = inputs.T
        for i in range(self.len_nets):
            activ = forward_layer(self.nets[i], activ, self.biases[i],
                                  self.activ_funcs[i])
        return activ.T

    def convert_to_onehot(self, array: array) -> array:
        """Convert the category to onehot encoding format, to adapt
        softmax multi categories classification. array ->
        (batchsize, category)"""

        batch_size = array.shape[0]
        num_classes = self.net_shape[-1]
        onehot = np.zeros((batch_size, num_classes))
        onehotindex = array[:, 0].astype(int)
        onehot[np.arange(batch_size), onehotindex] = 1
        array = onehot
        return array

    def train(self, inputs: array, truths: array, max_iter: int = 10000,
              learning_rate: float = 0.01, batch_size: int = 50,
              visualize: bool = True, test_ratio: float = 0.8,
              threshold: str | float = None, animation: str = None):
        """Train a dataset, it will first chunk dataset into mini-batches,
        if batch_size is 1, it will perfom SGD."""

        self.check_train_params(inputs, truths)
        self.prepare(visualize, threshold)
        (inputs_train, truths_train, inputs_test,
         truths_test) = split_dataset(inputs, truths, test_ratio)

        cce = "CategoricalCrossEntropy"
        if self.loss_func == cce and self.activ_funcs[-1] == softmax:
            truths_test = self.convert_to_onehot(truths_test)
            truths_train = self.convert_to_onehot(truths_train)
        startTime = datetime.now()
        try:
            self.show_record(0, inputs_train, inputs_test,
                             truths_train, truths_test, startTime, animation)
            for epoch in range(max_iter):
                inputs_train, truths_train = shuffle_data(inputs_train,
                                                          truths_train)
                # mini_batch_training
                count = 0
                while count < len(inputs_train):
                    inputs_batch = inputs_train[count: count + batch_size]
                    truths_batch = truths_train[count: count + batch_size]
                    self.train_batch(inputs_batch, truths_batch, learning_rate)
                    count += batch_size
                # mini_batch_training
                stop = self.show_record(epoch + 1, inputs_train,
                                        inputs_test, truths_train,
                                        truths_test, startTime, animation)
                if stop:
                    break
            print("[TRAINING DONE]")
            self.plt.ioff()
            self.plt.show()
            self.plt.close()
        except KeyboardInterrupt:
            print("Stopped by user.\033[?25h")
        self.save_weights()
        self.plt.close()

    def load_weights(self, file: str) -> None:
        """Load weights from a params.json file, in order to predict
        with a trained model, or continue the fine tuning."""

        if file is None:
            return
        with open(file, mode="r") as f:
            params = json.load(f)

        if len(params["shape"]) != len(self.net_shape):
            info = "Mismatched network with current configuration."
            raise ValueError(info)
        for i in range(len(params["shape"])):
            if params["shape"][i] != self.net_shape[i]:
                info = "Mismatched network with current configuration."
                raise ValueError(info)

        ws = [np.array(w) for w in params["weights"]]
        bs = [np.array(b) for b in params["biases"]]
        for i in range(len(self.nets)):
            if (
                self.nets[i].shape != ws[i].shape
                or self.biases[i].shape != bs[i].shape
            ):
                print("[Load params from file failed, mismatched]")
                return
        self.nets = ws
        self.biases = bs

    def save_weights(self) -> None:
        """Save training weights into a params.json file."""

        weights_li = [arr.tolist() for arr in self.nets]
        biases_li = [arr.tolist() for arr in self.biases]
        model_params = {
            "shape": self.net_shape,
            "weights": weights_li,
            "biases": biases_li,
        }
        with open("params.json", "w", encoding="utf-8") as f:
            json.dump(model_params, f, indent=4)
        print("[Params saved => (params.json)]\033[?25h")

    def test_classification(self, test_truths: array,
                            test_result: array) -> array:
        count = 0
        length = len(test_result)
        if self.loss_func == "CategoricalCrossEntropy":
            loss_test = loss(ce_loss,
                             self.convert_to_onehot(test_truths),
                             test_result)
            test_result = np.argmax(test_result, axis=1, keepdims=True)
            for i in range(len(test_result)):
                count += cmp_correct(test_result[i], test_truths[i], i)
        elif self.loss_func == "BinaryCrossEntropy":
            if test_result.shape[1] == 1:
                loss_test = loss(bce_loss, test_truths, test_result)
                test_result = (test_result > 0.5).astype(int)
                for i in range(len(test_result)):
                    count += cmp_correct(test_result[i], test_truths[i], i)
            elif test_result.shape[1] == 2:
                test_truths_onehot = self.convert_to_onehot(test_truths)
                loss_test = loss(bce_loss, test_truths_onehot, test_result)
                test_result_1d = np.argmax(test_result, axis=1, keepdims=True)
                for i in range(len(test_result)):
                    print(f"IDX {i}  TRUTH {test_truths_onehot[i]} ", end="")
                    print(f"=> PRED [{test_result[i][0]:.3f} ] ", end="")
                    print(f"{test_result[i][1]:.3f}] ", end="")
                    print(f"{test_result[i][1]:.3f}]", end="")
                    if test_result_1d[i] == test_truths[i]:
                        print("  \033[32mOK\033[0m")
                        count += 1
                    else:
                        print("  \033[31mKO\033[0m")
                test_result = test_result_1d
            else:
                info = "Binary Cross Entropy doesn't support more dimensions."
                raise ValueError(info)
        print(f"[Correct_Predict] [{count}/{length}]  ", end="")
        print(f"[Acc_Test] {(count / length) * 100:.2f}%  ", end="")
        print(f"[{self.loss_func}] {loss_test:.4f}")
        return test_result

    def test(self, test_inputs: array, test_truths: array) -> None:
        """Test for a dataset, if is classification task, the program
        will check last layer activation function to determine if CCE or BCE
        is applied, and convert onehot encoding to actual category if softmax
        is provided."""

        test_result = self.inference(test_inputs)

        if self.classification:
            test_result = self.test_classification(test_truths, test_result)
        elif self.loss_func == "MeanSquareError":
            print(test_truths.shape, test_result.shape)
            loss_test = loss(mse_loss, test_truths, test_result)
            print(f"[Loss] {loss_test:.4f}")
        else:
            raise ValueError("Unsupported loss function")
        with open("predictions.json", "w", encoding="utf-8") as f:
            json.dump({"prediction": [r.tolist() for r in test_result]},
                      f, indent=4)
        print("[Predictions saved => (predictions.json)]\033[?25h")

    def test_animation(self, test_inputs: array,
                       test_truths: array, animation: str) -> None:
        """Test for a dataset, and show the animation."""

        test_result = self.inference(test_inputs)
        self.plt.clf()
        inputs = test_inputs[:, 0].flatten()
        truths = np.array(test_truths)[:, 0].flatten()
        outputs = np.array(test_result)[:, 0].flatten()

        sorted_index = np.argsort(inputs)
        inputs_sorted = inputs[sorted_index]
        truths_sorted = truths[sorted_index]
        outputs_sorted = outputs[sorted_index]

        self.plt.scatter(inputs_sorted, truths_sorted,
                         c="blue", label="Truth", s=10)
        if animation == "plot":
            self.plt.plot(inputs_sorted, outputs_sorted,
                          c="red", label="Prediction", lw=1)
        elif animation == "scatter":
            self.plt.scatter(inputs_sorted, outputs_sorted,
                             c="red", label="Prediction", s=10)
        elif animation is None:
            pass
        else:
            raise TypeError("Wrong animation type")
        self.plt.legend(loc="lower left")
        self.plt.pause(0.1)
        self.plt.clf()

    def cal_loss(self, truths_train: array, predicts_train: array,
                 truths_test: array, predicts_test: array) -> tuple:
        "Use raw value to calculate loss, no need to convert to category."

        loss_func = None
        if self.loss_func == "CategoricalCrossEntropy":
            loss_func = ce_loss
        elif self.loss_func == "BinaryCrossEntropy":
            loss_func = bce_loss
        elif self.loss_func == "MeanSquareError":
            loss_func = mse_loss
        else:
            raise ValueError("Not supported loss function")

        loss_train = loss(loss_func, truths_train, predicts_train)
        loss_test = loss(loss_func, truths_test, predicts_test)
        return loss_train, loss_test

    def get_cat_by_predict(self, pred_train: array,
                           pred_test: array,
                           truths_train: array,
                           truths_test: array) -> tuple:
        """Get category by the output prediction of neural network."""

        bce = "BinaryCrossEntropy"
        cce = "CategoricalCrossEntropy"
        if self.loss_func == cce and self.activ_funcs[-1] == softmax:
            predict_train_cat = pred_train.argmax(axis=1, keepdims=True)
            predict_test_cat = pred_test.argmax(axis=1, keepdims=True)
            truths_train_original = truths_train.argmax(axis=1, keepdims=True)
            truths_test_original = truths_test.argmax(axis=1, keepdims=True)
        elif self.loss_func == bce and self.activ_funcs[-1] == sigmoid:
            predict_train_cat = (pred_train >= 0.5).astype(np.int32)
            predict_test_cat = (pred_test >= 0.5).astype(np.int32)
            truths_train_original = truths_train
            truths_test_original = truths_test
        else:
            info = "Not supported combination of loss and activation func."
            raise ValueError(info)

        acc_train = accuracy_1d(truths_train_original, predict_train_cat)
        acc_test = accuracy_1d(truths_test_original, predict_test_cat)
        return acc_train, acc_test

    def collect_train_record(self, epoch: int, loss_train: float,
                             loss_test: float, acc_train: float,
                             acc_test: float) -> None:
        """Collect training history to show and draw the plots."""

        self.graph_loss_train.append(loss_train)
        self.graph_loss_test.append(loss_test)
        self.graph_epoch.append(epoch)
        if self.classification and acc_test and acc_train:
            self.graph_acc_train.append(acc_train)
            self.graph_acc_test.append(acc_test)

    def show_train_info(self, epoch: int, startTime: datetime,
                        loss_train: float, loss_test: float,
                        acc_train: float, acc_test: float) -> None:
        """Show training info during the training process."""

        if epoch % 10 == 0:
            time = str(datetime.now() - startTime).split(".")[0]
            if not self.classification:
                print(f"\033[?25l[EPOCH] {epoch}  ", end="")
                print(f"[Loss_Train] {loss_train:.4f} ", end="")
                print(f"[Loss_Val] {loss_test:.4f}  ", end="")
                print(f"[TIME] {time}\033[?25h")
            else:
                print(f"\033[?25l[EPOCH] {epoch}  ", end="")
                print(f"[Loss_Train] {loss_train:.4f} ", end="")
                print(f"[Loss_Val] {loss_test:.4f}  ", end="")
                print(f"[Acc_Train] {(acc_train * 100):.1f}% ", end="")
                print(f"[Acc_Val] {(acc_test * 100):.1f}% ", end="")
                print(f"[TIME] {time}\033[?25h")

    def show_record(self, epoch: int, inputs_train: array,
                    inputs_test: array, truths_train: array,
                    truths_test: array, startTime: datetime,
                    animation: str) -> bool:
        """Show and record the loss, return a boolean to determine
        if training continue."""

        if epoch % 1 == 0:
            pred_train = self.inference(inputs_train)
            pred_test = self.inference(inputs_test)

            loss_train, loss_test = self.cal_loss(truths_train,
                                                  pred_train,
                                                  truths_test,
                                                  pred_test)
            acc_train, acc_test = None, None

            if self.classification:
                acc_train, acc_test = self.get_cat_by_predict(pred_train,
                                                              pred_test,
                                                              truths_train,
                                                              truths_test)

            self.collect_train_record(epoch, loss_train,
                                      loss_test, acc_train, acc_test)
            if animation != "none" and epoch % 50 == 0:
                self.test_animation(inputs_test[:50], truths_test[:50],
                                    animation)
            self.show_train_info(epoch, startTime, loss_train, loss_test,
                                 acc_train, acc_test)

            # Early stop --------------------------------------
            if self.loss_train is not None and self.loss_test is not None:
                if abs(self.loss_train - loss_train) < self.loss_threshold:
                    return True
            # Early stop --------------------------------------
            self.loss_train = loss_train
            self.loss_test = loss_test
        return False

    def save_plots(self):
        """Show loss func plots and if classification is applied
        show also accuracy."""

        plt.plot(self.graph_epoch, self.graph_loss_train,
                 c="cyan", lw=1, label="Training loss")
        plt.plot(self.graph_epoch, self.graph_loss_test,
                 c="orange", linestyle="--", lw=1, label="Test loss")
        plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
        plt.title("Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig("visualize/loss.png", dpi=300, bbox_inches='tight')
        plt.close()

        if self.classification:
            plt.plot(self.graph_epoch, self.graph_acc_train,
                     c="cyan", lw=1, label="Training accuracy")
            plt.plot(self.graph_epoch, self.graph_acc_test,
                     c="orange", linestyle="--", lw=1, label="Test accuracy")
            plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
            plt.title("Learning Curves")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc="lower right")
            plt.savefig("visualize/accuracy.png", dpi=300, bbox_inches='tight')
            plt.close()

    def show_test_img(self, test_inputs: array, shape: tuple,
                      index: int) -> None:
        """If test data is image, show them"""

        print(f"Show test image {index} with shape {shape}")
        img = test_inputs[index].reshape(shape)
        plt.imshow(img, cmap="gray")
        plt.show()
        plt.close()

    def prepare(self, visualize: bool, threshold: float) -> None:
        """Create folder to save training result."""

        if visualize:
            os.makedirs("visualize", exist_ok=True)
            plt.ion()
        if threshold is not None:
            self.loss_threshold = threshold
