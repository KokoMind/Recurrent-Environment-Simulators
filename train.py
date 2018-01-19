import numpy as np
import tensorflow as tf
from tqdm import tqdm
from base.base_train import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, sess, model, data, config):
        super(Trainer, self).__init__(sess, model, data, config, None)

    def train(self):

        initial_lstm_state = np.zeros((2, self.config.batch_size, self.config.lstm_size))

        for cur_epoch in range(self.cur_epoch_tensor.eval(self.sess), self.config.n_epochs + 1, 1):
            cur_iterations = 0
            losses = []
            cur_epoch = self.cur_epoch_tensor.eval(self.sess)
            loop = tqdm(self.data.next_batch(), total=self.config.nit_epoch, desc="epoch-" + str(cur_epoch) + "-")
            cur_it = self.global_step_tensor.eval(self.sess)

            for batch_x, batch_y, batch_actions, batch_rewards, new_sequence in loop:
                # Update the Global step
                self.global_step_assign_op.eval(session=self.sess, feed_dict={
                    self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})

                if new_sequence:
                    feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.actions: batch_actions,
                                 self.model.rewards: batch_rewards,
                                 self.model.initial_lstm_state: initial_lstm_state, self.model.is_training: True}
                    last_state = self.sess.run(self.model.final_lstm_state, feed_dict)
                else:
                    feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.actions: batch_actions,
                                 self.model.rewards: batch_rewards,
                                 self.model.initial_lstm_state: last_state, self.model.is_training: True}
                    output_softmax, _, loss, last_state = self.sess.run(
                        [self.model.output_softmax, self.model.train_step, self.model.loss,
                         self.model.final_lstm_state], feed_dict)

                    losses.append(loss)

                cur_iterations += 1
                # finish the epoch
                if cur_iterations >= self.config.nit_epoch:
                    break
            loss = np.mean(losses)

            summaries_dict = {'loss': loss}
            self.add_scaler_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=self.model.summaries)

            loop.close()
            print("epoch-" + str(cur_epoch) + "-" + "loss-" + str(loss))

            # increment_epoch
            self.cur_epoch_assign_op.eval(session=self.sess,
                                          feed_dict={self.cur_epoch_input: self.cur_epoch_tensor.eval(self.sess) + 1})

            # Save the current checkpoint
            if cur_epoch % self.config.save_every == 0:
                self.save()

            if cur_epoch % self.config.test_every == 0:
                self.test(cur_it)

        print("Training Finished")

    def test(self, cur_it,type='test'):
        x, a = self.data.sample(type)
        lstm_state = np.zeros((2, self.config.batch_size, self.config.lstm_size))
        out = x[:, 0]
        # test_images = np.zeros([self.config.test_steps,self.config.batch_size] + self.config.state_size)

        feed_dict = {self.model.x_test: out, self.model.actions_test: a[:,0],
                     self.model.initial_lstm_state_test: lstm_state, self.model.is_training: False}
        lstm_state = self.sess.run(self.model.lstm_state_test, feed_dict)

        for i in range(1, self.config.test_steps):
            feed_dict = {self.model.x_test: out, self.model.actions_test: a[:,i],
                         self.model.initial_lstm_state_test: lstm_state, self.model.is_training: False}
            out, lstm_state = self.sess.run([self.model.output_softmax_test, self.model.lstm_state_test], feed_dict)

            test_images = np.concatenate((x[:,i], out),axis=2)
            summaries_dict = {'image'+str(i): test_images[...,1,None]}
            self.add_image_summary(cur_it, summaries_dict=summaries_dict,scope=type)
