from neural_network import *
import numpy as np


class DQN(NN):

    def __init__(self, shape: list, activation_functions: list, init_methods:list, classification:bool=False, loss:str="MeanSquareError"):
        super().__init__(shape, activation_functions, init_methods, classification, loss)
    
    
    def train_batch_rl(self, inputs, target, learning_rate=0.01):
        """Train a batch."""

        inputs_batch = inputs.T   # (batch_size, features) -> (features, batch_size)
        # -----------------------------forward --------------------------------
        actives = [inputs_batch]
        Bgrads = []
        Wgrads = []
        for i in range(self.len_nets):
            actives.append(forward_layer(self.nets[i], actives[i], self.biases[i], self.activ_funcs[i]))
        # -----------------------------forward end-----------------------------
        # -----------------------------back probab --------------------------------
        # Last layer
        # print(actives[-1].T[0])
        target_idx = actives[-1].argmax(axis=0, keepdims=True)[0]

        Q = np.zeros_like(actives[-1])
        Q[target_idx] = actives[-1][target_idx] - target
        
        if self.loss_func == "MeanSquareError":
            local_grad = Q * activ_deriv(self.activ_funcs[-1], actives[-1], self.deriv_map) # last layer difference
        elif self.loss_func == "CategoricalCrossEntropy": # and self.activ_funcs[-1] == softmax:
            local_grad = Q
        else:
            print(self.loss_func, self.activ_funcs[-1])
            raise ValueError("Wrong loss function")
        
        Bgrads.append(np.mean(local_grad, axis=1, keepdims=True))
        Wgrads.append(local_grad @ actives[-2].T / 1)
    
        for i in range(self.len_nets - 1, 0, -1):
            loss_prev_layer = self.nets[i].T @ local_grad  #cal the loss of prev layer
            local_grad = loss_prev_layer * activ_deriv(self.activ_funcs[i - 1], actives[i], self.deriv_map) 
            Bgrads.append(np.mean(local_grad, axis=1, keepdims=True))
            Wgrads.append(local_grad @ actives[i - 1].T / 1)
        # -----------------------------back probab end-----------------------------
        gradient_descent(self.nets, self.biases, Wgrads[::-1], Bgrads[::-1], learning_rate)

        


def dqn_train(nn, conf, inputs, truths):
    nn.train(inputs, truths, 
             conf["max_epoch"], 
             conf["learning_rate"], 
             batch_size=conf["batch_size"], 
             test_ratio=conf["train_ratio"],
             threshold=conf["threshold"],
             animation=conf["animation"])
    nn.save_plots()


def create_dqn():
    conf = {
        "shape": [100, 50, 15, 10, 4],
        "activation_funcs": ["relu", "relu", "relu", "softmax"],
        "weights_init": ["he", "he", "he", "xavier"],
        "loss": "CategoricalCrossEntropy",
        "max_epoch": 10000,
        "learning_rate": 0.01,
        "batch_size": 50,
        "classification": True,
        "animation": "plot",
        "train_ratio": 1,
        "threshold": False,
        "index": True
    }

    dqn = DQN(conf["shape"], get_activation_funcs_by_name(conf["activation_funcs"]), 
            conf["weights_init"],
            classification=conf["classification"],
            loss=conf["loss"])
    return dqn, conf

dqn, conf = create_dqn()