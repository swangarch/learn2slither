from neural_network import get_activation_funcs_by_name, NN
from numpy import ndarray as array
import json


class Model(NN):
    def __init__(self, shape, activation_functions: list[callable],
                 init_methods: list[str], classification: bool = False,
                 loss: str = "MeanSquareError"):
        super().__init__(shape, activation_functions, init_methods,
                         classification, loss)

    def train_batch_rl(self, epoch: int, inputs: array,
                       truths: array, learning_rate: float = 0.01) -> None:
        """Train a batch, the inputs and truths have to be already chunked
        into batch. This function will perform feed foward, back probagation,
        and gradient descent, the process to train the model."""
        self.train_batch(inputs, truths, learning_rate)

    def sync_params(self, main_model: "Model") -> None:
        """Synchronize parameters, copy their weights to target model"""
        self.nets = [weights.copy() for weights in main_model.nets]
        self.biases = [bias.copy() for bias in main_model.biases]

    def save_weights_to(self, path: str) -> None:
        """Save training weights into a path of json file."""

        weights_li = [arr.tolist() for arr in self.nets]
        biases_li = [arr.tolist() for arr in self.biases]
        model_params = {
            "shape": self.net_shape,
            "weights": weights_li,
            "biases": biases_li,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_params, f, indent=4)
        print(f"[Params saved => ({path})]\033[?25h")

    @classmethod
    def create_model(cls) -> "Model":
        """Create a model for this DQN."""
        conf = {
            "shape": [20, 128, 64, 16, 4],
            "activation_funcs": ["leaky_relu", "leaky_relu",
                                 "leaky_relu", "none"],
            "weights_init": ["he", "he", "he", "he"],
            "loss": "MeanSquareError",
            "classification": False,
            "animation": "plot",
            "threshold": False,
            "index": True
        }
        model = Model(conf["shape"], get_activation_funcs_by_name(
                    conf["activation_funcs"]),
                    conf["weights_init"],
                    classification=conf["classification"],
                    loss=conf["loss"]
                )
        return model

    @classmethod
    def copy(cls, other: "Model") -> "Model":
        c = cls(other.net_shape, other.activ_funcs, other.init_methods)
        c.sync_params(other)
        return c
