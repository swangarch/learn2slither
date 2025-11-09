from neural_network import *
from numpy import ndarray as array
import os


class Model(NN):
    def __init__(self, shape, activation_functions, init_methods, classification = False, loss = "MeanSquareError"):
        super().__init__(shape, activation_functions, init_methods, classification, loss)

        self.plt.ion()
        os.makedirs("visualize", exist_ok=True)


    def train_batch_rl(self, epoch, inputs:array, truths:array, learning_rate:float=0.01) -> None:
        """Train a batch, the inputs and truths have to be already chunked into batch.
        This function will perform feed foward, back probagation, and gradient descent,
        the process to train the model, this function will visualize loss curve.
        """
        self.train_batch(inputs, truths, learning_rate)    
        if epoch % 5000 == 0:
            loss_train = loss(mse_loss, truths.T, self.inference(inputs).T)
            self.graph_loss_train.append(loss_train)
            self.graph_epoch.append(epoch)
            self.plt.clf()
            if len(self.graph_epoch) > 0 and len(self.graph_loss_train) > 0:
                self.plt.plot(self.graph_epoch, self.graph_loss_train, c="red", label="Loss", lw=1)
                self.plt.legend(loc="lower left")
                self.plt.xlabel("Move")
                self.plt.ylabel("Loss")
                self.plt.title("Training loss curve")
            self.plt.pause(0.01)


    def close_visual(self):
        self.plt.ioff()
        self.plt.show()
        self.plt.close()


    def save_plot(self):
        """Show loss func plots and if classification is applied show also accuracy."""
        self.plt.plot(self.graph_epoch, self.graph_loss_train, c="red", lw=1, label="Loss")
        self.plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
        self.plt.title("Loss Curves")
        self.plt.xlabel("Epochs")
        self.plt.ylabel("Loss")
        self.plt.legend(loc="upper right")
        self.plt.savefig("visualize/loss.png", dpi=300, bbox_inches='tight')
        self.plt.close()


    @classmethod
    def create_model(cls):
        conf = {
            "shape": [20, 128, 64, 16, 4],
            "activation_funcs": ["leaky_relu", "leaky_relu", "leaky_relu", "none"],
            "weights_init": ["he", "he", "he", "he"],
            "loss": "MeanSquareError",
            "classification": False,
            "animation": "plot",
            "threshold": False,
            "index": True
        }
        model = Model(conf["shape"], get_activation_funcs_by_name(conf["activation_funcs"]), 
                    conf["weights_init"],
                    classification=conf["classification"],
                    loss=conf["loss"]
                )
        return model