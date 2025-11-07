from neural_network import *
import numpy as np
from numpy import ndarray as array


class DQN(NN):
    def __init__(self, shape, activation_functions, init_methods, classification = False, loss = "MeanSquareError"):
        super().__init__(shape, activation_functions, init_methods, classification, loss)

        self.plt.ion()

    def train_batch_rl(self, epoch, inputs:array, truths:array, learning_rate:float=0.01) -> None:
        """Train a batch, the inputs and truths have to be already chunked into batch.
        This function will perform feed foward, back probagation, and gradient descent,
        the process to train the model.
        """
        self.train_batch(inputs, truths, learning_rate)    
        if epoch % 500 == 0:
            loss_train = loss(mse_loss, truths.T, self.inference(inputs).T)
            self.graph_loss_train.append(loss_train)
            self.graph_epoch.append(epoch)
            self.plt.clf()
            self.plt.plot(self.graph_epoch, self.graph_loss_train, c="red", label="Loss", lw=1)
            self.plt.legend(loc="lower left")
            self.plt.xlabel("Move")
            self.plt.ylabel("Loss")
            self.plt.pause(0.01)

    def close_visual(self):
        self.plt.ioff()
        self.plt.show()
        self.plt.close()

    @classmethod
    def create_dqn(cls):
        conf = {
            "shape": [20, 128, 64, 16, 4],
            "activation_funcs": ["leaky_relu", "leaky_relu", "leaky_relu", "none"],
            "weights_init": ["he", "he", "he", "he"],
            "loss": "MeanSquareError",
            "classification": True,
            "animation": "plot",
            "threshold": False,
            "index": True
        }
        dqn = DQN(conf["shape"], get_activation_funcs_by_name(conf["activation_funcs"]), 
                    conf["weights_init"],
                    classification=conf["classification"],
                    loss=conf["loss"]
                )
        return dqn