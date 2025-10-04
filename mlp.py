#!/usr/bin/python3

from neural_network import NN,  get_activation_funcs_by_name
import sys


def mlp_train(nn, conf, inputs, truths):
    nn.train(inputs, truths, 
             conf["max_epoch"], 
             conf["learning_rate"], 
             batch_size=conf["batch_size"], 
             test_ratio=conf["train_ratio"],
             threshold=conf["threshold"],
             animation=conf["animation"])
    nn.save_plots()


def mlp_create_nn():
    conf = {
        "shape": [100, 15, 10, 4],
        "activation_funcs": ["relu", "relu", "sigmoid"],
        "weights_init": ["xavier", "xavier", "xavier"],
        "loss": "MeanSquareError",
        "max_epoch": 10000,
        "learning_rate": 0.005,
        "batch_size": 1,
        "classification": True,
        "animation": "none",
        "train_ratio": 1,
        "threshold": False,
        "index": True
    }

    nn = NN(conf["shape"], get_activation_funcs_by_name(conf["activation_funcs"]), 
            conf["weights_init"],
            classification=conf["classification"],
            loss=conf["loss"])
    return nn, conf

nn, conf = mlp_create_nn()

# def main():
#     try:
#         argv = sys.argv
#         if len(argv) == 1 or ((len(argv) == 2 and argv[1] == "--help")):
#             print_help()
#         elif argv[1] == "-s" and len(argv) == 4:
#             mlp_splitdata(argv[2], argv[3], 132, 0.8)
#         elif len(argv) == 4 or len(argv) == 5:
#             nn, inputs, truths, conf = mlp_create_nn(argv)
#             if argv[1] == "-t":
#                 mlp_train(nn, conf, inputs, truths)
#             elif argv[1] == "-p":
#                 nn.test(inputs, truths)
#             else:
#                 raise ValueError("Wrong arguments. Try: python mlp.py --help")
#         else:
#             raise ValueError("Wrong arguments. Try: python mlp.py --help")

#     except KeyboardInterrupt as e:
#         print()
#         print("Stopped by user.\033[?25h")

#     except Exception as e:
#         print("Error:", e)


# if __name__ == "__main__":
#     main()