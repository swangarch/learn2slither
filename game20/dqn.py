from neural_network import *
import numpy as np


class DQN(NN):

    def __init__(self, shape: list, activation_functions: list, init_methods:list, classification:bool=False, loss:str="MeanSquareError"):
        super().__init__(shape, activation_functions, init_methods, classification, loss)

        # self.plt.ion()
        # self.loss_train = []
        # self.iter = []
    
    def train_batch_rl(self, inputs, target, learning_rate=0.01, batch_size=100):
        """Train a batch."""

        # -----------------------------forward --------------------------------
        
        total_samples = inputs.shape[0]
    
    # 如果样本数少于batch_size,使用全部样本
        if total_samples <= batch_size:
            sampled_inputs = inputs
            sampled_target = target
        else:
            # 随机抽取batch_size个样本
            indices = np.random.choice(total_samples, batch_size, replace=False)
            sampled_inputs = inputs[indices]
            sampled_target = target[indices]
            
        inputs_T = sampled_inputs.T
        target_T = sampled_target.T

        l = len(inputs)

        actives = [inputs_T]
        Bgrads = []
        Wgrads = []
        for i in range(self.len_nets):
            actives.append(forward_layer(self.nets[i], actives[i], self.biases[i], self.activ_funcs[i]))
        # -----------------------------forward end-----------------------------
        # -----------------------------back probab --------------------------------
        
        Q = np.zeros_like(actives[-1])
        Q = actives[-1] - target_T
        
        if self.loss_func == "MeanSquareError":
            local_grad = Q * activ_deriv(self.activ_funcs[-1], actives[-1], self.deriv_map) # last layer difference
        else:
            print(self.loss_func, self.activ_funcs[-1])
            raise ValueError("Wrong loss function")
        
        Bgrads.append(np.mean(local_grad, axis=1, keepdims=True))
        Wgrads.append(local_grad @ actives[-2].T / l)
    
        for i in range(self.len_nets - 1, 0, -1):
            loss_prev_layer = self.nets[i].T @ local_grad  #cal the loss of prev layer
            local_grad = loss_prev_layer * activ_deriv(self.activ_funcs[i - 1], actives[i], self.deriv_map) 
            Bgrads.append(np.mean(local_grad, axis=1, keepdims=True))
            Wgrads.append(local_grad @ actives[i - 1].T / l)
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
    nn.save_weights()


def create_dqn():
    conf = {
        "shape": [20, 30, 10, 4],
        "activation_funcs": ["relu", "relu", "none"],
        "weights_init": ["he", "he", "he"],
        "loss": "MeanSquareError",
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