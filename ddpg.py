__author__ = 'witwolf'

import tensorflow as tf
from collections import deque
import numpy as np
import random
from OUNoise import OUNoise


def random_init(shape):
    v = 1 / np.sqrt(shape[0])
    return tf.random_uniform(shape, minval=-v, maxval=v)


class Ddpg(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 p_learning_rate=0.0002,
                 q_learning_rate=0.001,
                 gamma=0.9,
                 eta=0.0003,
                 batch_size=64,
                 replay_buffer_size=1024 * 1024,
                 min_train_replays=1024 * 16,
                 logdir='',
                 save_path='',
                 *args,
                 **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hl1_dim = 250  # hidden layer 1
        self.hl2_dim = 250  # hidden layer 2

        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.min_train_replays = min_train_replays

        self.noise = OUNoise(action_dim)
        self.time_step = 0
        self.replay_buffer = deque()

        self.gamma = gamma
        self.eta = eta
        self.alpha = self.initial_alpha = 1.0
        self.final_alpha = 0.01
        self.p_learning_rate = p_learning_rate
        self.q_learning_rate = q_learning_rate

        self.save_path = save_path

        self.create_network()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        self.session.run(self.init_target_theta)
        # self.load()
        self.summary_writer = tf.train.SummaryWriter(logdir, self.session.graph)

    def theta_p(self):
        with tf.variable_scope("theta_p"):
            return [
                tf.Variable(random_init([self.state_dim, self.hl1_dim]), name="W1"),
                tf.Variable(random_init([self.hl1_dim]), name="b1"),
                tf.Variable(random_init([self.hl1_dim, self.hl2_dim]), name="W2"),
                tf.Variable(random_init([self.hl2_dim]), name="b2"),
                tf.Variable(random_init([self.hl2_dim, self.action_dim]), name="W3"),
                tf.Variable(random_init([self.action_dim]), name="b3")
            ]

    def theta_q(self):
        with tf.variable_scope("theta_q"):
            return [
                tf.Variable(random_init([self.state_dim, self.hl1_dim]), name='W1'),
                tf.Variable(random_init([self.hl1_dim]), name='b1'),
                tf.Variable(random_init([self.hl1_dim + self.action_dim, self.hl2_dim]), name='W2'),
                tf.Variable(random_init([self.hl2_dim]), name='b2'),
                tf.Variable(random_init([self.hl2_dim, 1]), name='W3'),
                tf.Variable(random_init([1]), name='b3')
            ]

    def create_policy_network(self, state, theta, name="policy_network"):
        with tf.variable_op_scope([state], name, name):
            h0 = tf.identity(state, "state")
            h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
            h2 = tf.nn.relu(tf.matmul(h1, theta[2]) + theta[3], name="h2")
            h3 = tf.identity(tf.matmul(h2, theta[4]) + theta[5], name='h3')
            action = tf.nn.tanh(h3, name='action')
            return action

    def create_q_network(self, state, action, theta, name='q_network'):
        with tf.variable_op_scope([state, action], name, name):
            h0 = tf.identity(state, name='state')
            h1_state = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1])
            # h1 = concat(h1_state,action)
            h1 = tf.concat(1, [h1_state, action], name="h1")
            h2 = tf.nn.relu(tf.matmul(h1, theta[2]) + theta[3], name="h2")
            h3 = tf.add(tf.matmul(h2, theta[4]), theta[5], name='h3')
            q = tf.squeeze(h3, [1], name='q')
            return q

    def create_network(self):

        theta_q, theta_p = self.theta_q(), self.theta_p()
        target_theta_q, target_theta_p = self.theta_q(), self.theta_p()

        # init target theta with the  same value of theta
        init_target_theta_q = [
            target_theta_q[i].assign(theta_q[i].value()) for i in range(len(theta_q))
            ]

        init_target_theta_p = [
            target_theta_p[i].assign(theta_p[i].value()) for i in range(len(theta_p))
            ]
        self.init_target_theta = init_target_theta_q + init_target_theta_p

        self.state = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
        self.action = tf.placeholder(tf.float32, [None, self.action_dim], 'action')
        self.next_state = tf.placeholder(tf.float32, [None, self.state_dim], 'next_state')
        self.reward = tf.placeholder(tf.float32, [None], 'reward')
        self.terminate = tf.placeholder(tf.bool, [None], 'terminate')

        #  q optimizer
        q = self.create_q_network(self.state, self.action, theta_q)
        next_action = self.create_policy_network(self.next_state, target_theta_p)
        next_q = self.create_q_network(self.next_state, next_action, target_theta_q)
        y_input = tf.stop_gradient(tf.select(self.terminate, self.reward, self.reward + self.gamma * next_q))
        q_error = tf.reduce_mean(tf.square(y_input - q))
        ## normalize
        q_loss = q_error + tf.add_n([0.01 * tf.nn.l2_loss(var) for var in theta_q])
        q_optimizer = tf.train.AdamOptimizer(self.q_learning_rate)
        grads_and_vars_q = q_optimizer.compute_gradients(q_loss, var_list=theta_q)
        q_train = q_optimizer.apply_gradients(grads_and_vars_q)

        #  policy optimizer
        self.action_exploration = self.create_policy_network(self.state, theta_p)
        q1 = self.create_q_network(self.state, self.action_exploration, theta_q)
        p_error = - tf.reduce_mean(q1)
        ## normalize
        p_loss = p_error + tf.add_n([0.01 * tf.nn.l2_loss(var) for var in theta_p])
        p_optimizer = tf.train.AdamOptimizer(self.p_learning_rate)
        grads_and_vars_p = p_optimizer.compute_gradients(p_loss, var_list=theta_p)
        p_train = p_optimizer.apply_gradients(grads_and_vars_p)

        # train q and update target_theta_q
        update_theta_q = [
            target_theta_q[i].assign(theta_q[i].value() * self.eta + target_theta_q[i].value() * (1 - self.eta)) for i
            in range(len(theta_q))]

        with tf.control_dependencies([q_train]):
            self.train_q = tf.group(*update_theta_q)

        # train p and update target_theta_p
        update_theta_p = [
            target_theta_p[i].assign(theta_p[i].value() * self.eta + target_theta_p[i].value() * (1 - self.eta)) for i
            in range(len(theta_p))]

        with tf.control_dependencies([p_train]):
            self.train_p = tf.group(*update_theta_p)


        # summary
        tf.scalar_summary('q_loss', q_loss)
        tf.scalar_summary('p_loss', p_loss)
        self.merged_op = tf.merge_all_summaries()

    def train(self):
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [v[0] for v in minibatch]
        action_batch = [v[1] for v in minibatch]
        reward_batch = [v[2] for v in minibatch]
        next_state_batch = [v[3] for v in minibatch]
        terminate_batch = [v[4] for v in minibatch]

        _, _, summary_str = self.session.run([self.train_p, self.train_q, self.merged_op], feed_dict={
            self.state: state_batch,
            self.action: action_batch,
            self.reward: reward_batch,
            self.terminate: terminate_batch,
            self.next_state: next_state_batch
        })
        self.summary_writer.add_summary(summary_str, self.time_step)
        self.summary_writer.flush()

        if self.time_step % 1000 == 0:
            self.save(self.time_step)

    def observe_action(self, state, action, reward, next_state, terminate):
        self.time_step += 1
        self.replay_buffer.append((state, action, reward, next_state, terminate))
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.popleft()

        if self.time_step > self.min_train_replays:
            self.train()

        if terminate:
            self.noise.reset()

    def exploration(self, state):
        action = self.session.run(self.action_exploration, feed_dict={self.state: [state]})[0]
        return np.clip(action, -1, 1)

    def exploration_with_noise(self, state):
        action = self.session.run(self.action_exploration, feed_dict={self.state: [state]})[0]
        self.alpha -= (self.initial_alpha - self.final_alpha) / 100000
        self.alpha = max(self.alpha, 0.0)
        noise = self.noise.noise() * self.alpha
        return np.clip(action + noise, -1, 1)

    def save(self, step):
        saver = tf.train.Saver()
        saver.save(self.session, save_path=self.save_path, global_step=step)

    def load(self):
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.session, checkpoint.model_checkpoint_path)
