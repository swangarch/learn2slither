import numpy as np
from game import IGame
import random as rd
from neural_network import NN
from .Model import Model
from collections import deque


class DQN():
    def __init__(self, game: IGame, model: NN, weights: str = None,
                 train_mode: bool = True) -> None:
        """Initialize the Deep Q network, which can take a game which
        impemented IGame interface, and a neural network as parameters,
        if weights are provided in a file, the model will load the
        pretrained weights. If train_mode is true, the model will perform
        training otherwise only playing."""
        # game visual setting
        self.game = game
        # neural network
        self.model = model
        self.target_model = Model.copy(model)
        if weights is not None:
            self.model.load_weights(weights)
        # memory pool
        self.memlen_max = 50000
        (self.states, self.actions, self.rewards,
         self.after_states, self.dones) = init_mem_pool(self.memlen_max)
        # hyper parameters
        self.decay = 0.9999
        self.gamma = 0.96
        self.batch_size = 64
        self.learning_rate = 0.002
        # self.decay = 0.9995
        # self.gamma = 0.98
        # self.batch_size = 128
        # self.learning_rate = 0.001
        if not train_mode:
            self.epsilon = 0.001
            self.min_explo_rate = 0.001
        else:
            self.epsilon = 1
            self.min_explo_rate = 0.05
        # train mode
        self.train_mode = train_mode

    def run(self, max_iter: int = 10000,
            running: bool = True, save: str = "params.json") -> None:
        """Run the main loop, int which we will update states, save states,
        actions rewards in a memory pool, then we use a neural network to
        perform the training."""
        session = 1
        action_count = 0
        try:
            while running and session < max_iter:
                running = self.reset_loop()
                state = self.game.build_state()
                action_idx = self.select_action(state)
                session += self.game.handle_step(action_idx)
                session = self.train_log(action_idx, session)
                if self.train_mode is True:
                    self.add_to_mem(state, action_idx, self.game.mem_num())
                    self.train(action_count)
                self.game.update_display(action_idx, session, self.epsilon,
                                         self.min_explo_rate, len(self.states))
                self.game.update_state()
                action_count += 1
        except KeyboardInterrupt:
            pass
        self.handle_quit(save)

    def reset_loop(self) -> bool:
        return self.game.reset()

    def handle_quit(self, path: str) -> None:
        """Save weights and cleaning ressources when quit."""
        self.game.quit_game()
        if self.train_mode is True:
            self.model.save_weights_to(path)

    def train(self, move_count: int) -> None:
        """Train the model by random sampling a batch from memory, calculate
        the target Q, then using neural network to train with current states
        and target Q."""
        if len(self.states) > 5000:
            # get a batch for training
            idxes = rd.sample(range(len(self.states)), min(self.batch_size,
                                                           len(self.states)))
            s_arr = np.array([self.states[i] for i in idxes]
                             ).squeeze(-1)
            next_s_arr = np.array([self.after_states[i] for i in idxes]
                                  ).squeeze(-1)
            actions_arr = np.array([self.actions[i] for i in idxes]
                                   ).reshape(-1, 1)
            rewards_arr = np.array([self.rewards[i] for i in idxes]
                                   ).reshape(-1, 1)
            dones_arr = np.array([self.dones[i] for i in idxes])

            Q_target = self.cal_Q_target(s_arr, next_s_arr,
                                         actions_arr, rewards_arr, dones_arr)
            self.model.train_batch_rl(move_count, s_arr,
                                      Q_target, self.learning_rate)
            if move_count % 100 == 0:
                self.target_model.sync_params(self.model)

    def cal_Q_target(self, states_array: deque[np.ndarray],
                     next_states_array: deque[np.ndarray],
                     actions_array: deque[int],
                     rewards_array: deque[float],
                     is_dones: deque[bool]) -> np.ndarray:
        """Calculate the Q target, using the state transfer equation,
        using model to predict Q value for current states and next states,
        let Q_curr(the quality of the choosen action) equals to current
        reward + Q_next(the quality of all next actions until the end of
        the game, we assume we will always choose the best actions in the
        future too). The Q_target will be the ideal Q value to pursue in the
        future actions."""
        Q_curr = self.model.inference(states_array)
        # shape: [N, 4]
        Q_next = self.target_model.inference(next_states_array)
        # shape: [N, 4]
        Q_target = Q_curr
        for i in range(len(actions_array)):
            action, r = actions_array[i], rewards_array[i]
            if is_dones[i] is True:
                Q_target[i, action[0]] = r
            else:
                Q_target[i, action[0]] = r + self.gamma * np.max(Q_next[i])
        return Q_target

    def add_to_mem(self, site_state: deque[np.ndarray],
                   currDirIdx: int, dup: int = 1) -> None:
        """Add to memory pool the information needed to calculated Q value
        and perform training, the goal is to leverage the experience of the
        past to learn the best action for a state. dup in the parameters means
        how many times the memory will be duplicated in the pool, shows how
        important this memory is."""
        for i in range(dup):
            self.states.append(site_state)
            self.actions.append(currDirIdx)
            self.rewards.append(self.game.reward)
            after_state = self.game.build_state()
            self.after_states.append(after_state)
            self.dones.append(self.game.done)

    def pred_action(self, site_state: deque[np.ndarray]) -> int:
        """Using model prediction to choose an action."""
        Q_curr = self.model.inference(site_state.T)
        predict_dir_idx = Q_curr.argmax(axis=1, keepdims=True)[0][0]
        return predict_dir_idx

    def select_action(self, site_state: deque[np.ndarray]) -> int:
        """Select action based on epsilon greedy strategy, either using a model
        prediction or taking a random action, this ratio will decay as the
        training session increases."""
        self.epsilon = max(self.min_explo_rate, self.epsilon * self.decay)
        if rd.random() > self.epsilon:
            return self.pred_action(site_state)
        else:
            return rd.randint(0, self.game.action_types - 1)

    def train_log(self, currDir: int, session: int) -> int:
        """Show the training log."""
        print(f"[SESSION] {session:4d} [ACTION] {currDir} ", end="")
        print(f"[REWARD] {self.game.reward:6.2f} ", end="")
        print(f"[MEM_LEN] {len(self.states):4d} {self.game.log_info(session)}")
        return session


def init_mem_pool(memlen_max: int) -> list[deque]:
    """Initialize the memory pools with deque, allows to have
    a fixed max length queque."""
    return [
        deque(maxlen=memlen_max),
        deque(maxlen=memlen_max),
        deque(maxlen=memlen_max),
        deque(maxlen=memlen_max),
        deque(maxlen=memlen_max)
    ]
