import numpy as np
import cv2
import matplotlib.pyplot as plt


class GenerateData:
    def __init__(self, config):
        """
        it just take the config file which contain all paths the generator needs
        :param config: configuration
        """

        self.config = config
        np.random.seed(2)
        x = np.load(config.states_path)
        self.rewards = np.load(config.rewards_path)
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        self.y = x[:, 1:]
        self.x = x[:, :-1]

        self.x = self.prepare_states(self.x)
        self.y = self.prepare_labels(self.y)

        # shuffles
        self.x = self.x[idx]
        self.y = self.y[idx]
        self.rewards = self.rewards[idx]
        self.prepare_actions(idx)
        self.rewards = np.expand_dims(self.rewards, axis=2)

        self.train_idx = int(self.config.train_ratio * self.x.shape[0])
        self.x_train = self.x[:self.train_idx]
        self.x_test = self.x[self.train_idx:]

        self.y_train = self.y[:self.train_idx]
        self.y_test = self.y[self.train_idx:]

        self.actions_train = self.actions[:self.train_idx]
        self.actions_test = self.actions[self.train_idx:]

        self.rewards_train = self.rewards[:self.train_idx]
        self.rewards_test = self.rewards[self.train_idx:]

    def next_batch(self):
        """
        :return: a tuple of all batches

        """
        while True:
            idx = np.random.choice(self.train_idx, self.config.batch_size)
            self.current_x = self.x_train[idx]
            self.current_y = self.y_train[idx]
            self.current_actions = self.actions_train[idx]
            self.current_rewards = self.rewards_train[idx]
            for i in range(0, self.config.episode_length, self.config.truncated_time_steps):
                if i == 0:
                    new_sequence = True
                else:
                    new_sequence = False
                batch_x = self.current_x[:, i:i + self.config.truncated_time_steps, :]
                batch_y = self.current_y[:, i:i + self.config.truncated_time_steps, :]
                batch_actions = self.current_actions[:, i:i + self.config.truncated_time_steps, :]
                batch_rewards = self.current_rewards[:, i:i + self.config.truncated_time_steps, :]
                yield batch_x, batch_y, batch_actions, batch_rewards, new_sequence

    def sample(self, type='train'):
        if type == 'train':
            idx = np.random.choice(self.x_train.shape[0], self.config.batch_size)
            return self.x_train[idx], self.actions_train[idx]

        elif type == 'test':
            idx = np.random.choice(self.x_test.shape[0], self.config.batch_size)
            return self.x_test[idx], self.actions_test[idx]

    def prepare_actions(self, idx):
        actions = np.load(self.config.actions_path)
        actions = np.expand_dims(actions, -1)
        self.actions = (np.arange(self.config.action_dim) == actions).astype(np.int32)
        self.actions = self.actions[idx]

    def prepare_states(self, x, env_id='Pong'):
        new_x = np.zeros((x.shape[0], x.shape[1], 96, 96, 1))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                retval2, threshold = cv2.threshold(x[i, j, :, :, 0].astype('uint8'), 89, 255, cv2.THRESH_BINARY)
                threshold = threshold.astype('uint8') // 255
                new_x[i, j, :, :, 0] = cv2.resize(threshold, (96, 96))
        new_x[:, :, :15, :, :] = 0

        # creating 2 channels
        new_x = (np.arange(2) == new_x).astype(int)

        return new_x

    def prepare_labels(self, x):
        new_x = np.zeros((x.shape[0], x.shape[1], 96, 96, 1))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                retval2, threshold = cv2.threshold(x[i, j, :, :, 0].astype('uint8'), 89, 255, cv2.THRESH_BINARY)
                threshold = threshold.astype('uint8') // 255
                new_x[i, j, :, :, 0] = cv2.resize(threshold, (96, 96))
        new_x[:, :, :15, :, :] = 0

        new_x = np.squeeze(new_x, -1)

        return new_x
