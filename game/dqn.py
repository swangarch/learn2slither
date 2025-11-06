from neural_network import *
import numpy as np
from numpy import ndarray as array


class DQN(NN):
    def train_batch_rl(self, epoch, inputs:array, truths:array, learning_rate:float=0.01) -> None:
        """Train a batch, the inputs and truths have to be already chunked into batch.
        This function will perform feed foward, back probagation, and gradient descent,
        the process to train the model.
        """

        self.train_batch(inputs, truths, learning_rate)
    
        if epoch % 100 == 0:
            loss_train = loss(mse_loss, truths.T, self.inference(inputs).T)
            self.graph_loss_train.append(loss_train)
            self.graph_epoch.append(epoch)
            self.plt.clf()
            self.plt.plot(self.graph_epoch, self.graph_loss_train, c="red", label="Loss", lw=1)
            self.plt.legend(loc="lower left")
            self.plt.xlabel("Move")
            self.plt.ylabel("Loss")
            self.plt.pause(0.01)


def create_dqn():
    conf = {
        "shape": [28, 64, 16, 4],
        "activation_funcs": ["leaky_relu", "leaky_relu", "none"],
        "weights_init": ["he", "he", "he"],
        "loss": "MeanSquareError",
        # "max_epoch": 10000,
        # "learning_rate": 0.01,
        # "batch_size": 50,
        "classification": True,
        "animation": "plot",
        "train_ratio": 0.95,
        "threshold": False,
        "index": True
    }

    dqn = DQN(conf["shape"], get_activation_funcs_by_name(conf["activation_funcs"]), 
            conf["weights_init"],
            classification=conf["classification"],
            loss=conf["loss"]
            )
    return dqn, conf