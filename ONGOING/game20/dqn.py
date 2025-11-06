from neural_network import *
import numpy as np


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
        "train_ratio": 1,
        "threshold": False,
        "index": True
    }

    dqn = NN(conf["shape"], get_activation_funcs_by_name(conf["activation_funcs"]), 
            conf["weights_init"],
            classification=conf["classification"],
            loss=conf["loss"]
            )
    return dqn, conf