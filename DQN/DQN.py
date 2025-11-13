import numpy as np
from game import *
import random as rd


class DQN():
    def __init__(self, game, model, weights=None, train_mode=True):
        
        # game visual setting
        self.game = game
        # neural network
        self.model = model
        if weights is not None:
            self.model.load_weights(weights)
        # memory pool
        self.memlen_max = 50000
        self.states, self.actions, self.rewards, self.after_states, self.dones = init_mem_pool(self.memlen_max)
        # hyper parameters
        self.decay = 0.9999
        self.gamma = 0.95
        self.batch_size = 64
        self.learning_rate = 0.002
        if train_mode == False:
            self.epsilon = 0.005
            self.min_explo_rate = 0.005
        else:
            self.epsilon = 1
            self.min_explo_rate = 0.05
        # train mode
        self.train_mode = train_mode


    def run(self, max_iter=10000, running=True):
        session = 0
        action_count = 0
        try:
            while running and session < max_iter:
                running = self.reset_loop()
                state = self.game.build_state()
                action_idx = self.select_action(state)
                session += self.game.handle_step(action_idx)
                if self.train_mode == True:
                    self.add_to_mem(state, action_idx, self.game.mem_num())
                    self.train(action_count)
                self.game.update_display(action_idx, session, self.epsilon, self.min_explo_rate, len(self.states))
                session = self.train_log(action_idx, session)
                self.game.update_state()
                action_count += 1
        except KeyboardInterrupt as e:
            pass
        self.handle_quit()


    def reset_loop(self):
        return self.game.reset()


    def handle_quit(self):
        self.game.quit_game()
        self.model.save_plot()
        self.model.close_visual()
        if self.train_mode == True:
            self.model.save_weights()


    def train(self, move_count):
        if len(self.states) > 5000:
            # get a batch for training
            indices = rd.sample(range(len(self.states)), min(self.batch_size, len(self.states)))
            states_array = np.array([self.states[i] for i in indices]).squeeze(-1)
            next_states_array = np.array([self.after_states[i] for i in indices]).squeeze(-1)
            actions_array = np.array([self.actions[i] for i in indices]).reshape(-1, 1)
            rewards_array = np.array([self.rewards[i] for i in indices]).reshape(-1, 1)
            dones_array = np.array([self.dones[i] for i in indices])

            Q_target = self.cal_Q_target(states_array, next_states_array, actions_array, rewards_array, dones_array)
            self.model.train_batch_rl(move_count, states_array, Q_target, self.learning_rate)


    def cal_Q_target(self, states_array, next_states_array, actions_array, rewards_array, is_dones):
        Q_curr = self.model.inference(states_array)  # shape: [N, 4]
        Q_next = self.model.inference(next_states_array)  # shape: [N, 4]
        Q_target = Q_curr
        for i in range(len(actions_array)):
            action, reward = actions_array[i], rewards_array[i]
            if is_dones[i] == True:
                Q_target[i, action[0]] = reward
            else:
                Q_target[i, action[0]] = reward + self.gamma * np.max(Q_next[i])
        return Q_target


    def add_to_mem(self, site_state, currDirIdx, dup=1):
        for i in range(dup):
            self.states.append(site_state)
            self.actions.append(currDirIdx)
            self.rewards.append(self.game.reward)
            after_state = self.game.build_state()
            self.after_states.append(after_state)
            self.dones.append(self.game.done)


    def pred_action(self, site_state):
        Q_curr = self.model.inference(site_state.T)
        predict_dir_idx = Q_curr.argmax(axis=1, keepdims=True)[0][0]
        return predict_dir_idx


    def select_action(self, site_state):
        self.epsilon = max(self.min_explo_rate, self.epsilon * self.decay)
        if rd.random() > self.epsilon:
            return self.pred_action(site_state)
        else:
            return rd.randint(0, 3)


    def train_log(self, currDir, session):
        print(f"[SESSION] {session:4d} [ACTION] {currDir} [REWARD] {self.game.reward:6.2f} [MEM_LEN] {len(self.states):4d} {self.game.log_info(session)}")
        return session