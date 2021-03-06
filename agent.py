from utils import to_bin
import time
import random
import numpy as np
from collections import deque
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

class Agent:
  """ DQN agent """
  def __init__(self, state_shape, actions):
    self.state_shape = state_shape
    self.actions = actions
    self.n_actions = len(self.actions)
    self.learning_rate=0.00001

    self.DQN = self.build_model()
    self.DQN_online = self.build_model()

  def build_model(self):
    model_input = Input(shape=(self.state_shape))

    x = Conv2D(32,   (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(64,   (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(64,   (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)

    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)

    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

    adv_stream = Flatten()(adv_stream)
    adv = Dense(self.n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean
    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(self.learning_rate), loss=tf.keras.losses.Huber())

    return model
    # """ Model builder function """
    # self.q_true = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
    # self.a_true = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
    # self.reward = tf.placeholder(dtype=tf.float32, shape=[], name='reward')
    # self.input = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.state_shape, name='input')
    # # self.input_float = tf.to_float(self.input) / 255.
    # # self.input = tf.convert_to_tensor(self.input[:,:,:3])

    # # Online network
    # with tf.variable_scope('online'):
    #     self.conv_1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=3, strides=4, activation=tf.nn.relu)
    #     self.conv_2 = tf.layers.conv2d(inputs=self.conv_1, filters=64, kernel_size=1, strides=2, activation=tf.nn.relu)
    #     self.conv_3 = tf.layers.conv2d(inputs=self.conv_2, filters=64, kernel_size=1, strides=1, activation=tf.nn.relu)
    #     self.flatten = tf.layers.flatten(inputs=self.conv_3)
    #     self.dense = tf.layers.dense(inputs=self.flatten, units=len(self.actions), activation=tf.nn.relu)
    #     self.output = tf.layers.dense(inputs=self.dense, units=len(self.actions), name='output')
    # # Target network
    # with tf.variable_scope('target'):
    #     self.conv_1_target = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=3, strides=4, activation=tf.nn.relu)
    #     self.conv_2_target = tf.layers.conv2d(inputs=self.conv_1_target, filters=64, kernel_size=1, strides=2, activation=tf.nn.relu)
    #     self.conv_3_target = tf.layers.conv2d(inputs=self.conv_2_target, filters=64, kernel_size=1, strides=1, activation=tf.nn.relu)
    #     self.flatten_target = tf.layers.flatten(inputs=self.conv_3_target)
    #     self.dense_target = tf.layers.dense(inputs=self.flatten_target, units=len(self.actions), activation=tf.nn.relu)
    #     self.output_target = tf.stop_gradient(tf.layers.dense(inputs=self.dense_target, units=len(self.actions), name='output_target'))
    # # Optimizer
    # self.action = tf.argmax(input=self.output, axis=1)
    # self.q_pred = tf.gather_nd(params=self.output, indices=tf.stack([tf.range(tf.shape(self.a_true)[0]), self.a_true], axis=1))
    # self.loss = tf.losses.huber_loss(labels=self.q_true, predictions=self.q_pred)
    # self.train = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(self.loss)
    # # Summaries
    # self.summaries = tf.summary.merge([
    #     tf.summary.scalar('reward', self.reward),
    #     tf.summary.scalar('loss', self.loss),
    #     tf.summary.scalar('max_q', tf.reduce_max(self.output))
    # ])
    # self.writer = tf.summary.FileWriter(logdir='./logs', graph=self.session.graph)

  def copy_model(self):
    """ Copy weights to target network """
    self.session.run([tf.assign(new, old) for (new, old) in zip(tf.trainable_variables('target'), tf.trainable_variables('online'))])

  def save_model(self):
    """ Saves current model to disk """
    self.saver.save(sess=self.session, save_path='./models/model', global_step=self.step)

  def add(self, experience):
    """ Add observation to experience """
    self.memory.append(experience)

  def predict(self, model, state):
    """ Prediction """
    if model == 'online':
        return self.session.run(fetches=self.output, feed_dict={self.input: np.array(state)})
    if model == 'target':
        return self.session.run(fetches=self.output_target, feed_dict={self.input: np.array(state)})

  def run(self, state):
    """ Perform action """
    if np.random.rand() < self.eps:
        # Random action
        action = self.actions[random.randint(0, len(self.actions) - 1)]
    else:
        # Policy action
        q = self.predict('online', np.expand_dims(state, 0))
        action = np.argmax(q)
        action = to_bin(action)
    # Decrease eps
    self.eps *= self.eps_decay
    self.eps = max(self.eps_min, self.eps)
    # Increment step
    self.step += 1
    return action

  def learn(self):
    """ Gradient descent """
    # Sync target network
    if self.step % self.copy == 0:
        self.copy_model()
    # Checkpoint model
    if self.step % self.save_each == 0:
        self.save_model()
    # Break if burn-in
    if self.step < self.burnin:
        return
    # Break if no training
    if self.learn_step < self.learn_each:
        self.learn_step += 1
        return
    # Sample batch
    batch = random.sample(self.memory, self.batch_size)
    state, next_state, action, reward, done, info = map(np.array, zip(*batch))
    # Get next q values from target network
    next_q = self.predict('target', next_state)
    # Calculate discounted future reward
    if self.double_q:
        q = self.predict('online', next_state)
        a = np.argmax(q, axis=1)
        target_q = reward + (1. - done) * self.gamma * next_q[np.arange(0, self.batch_size), a]
    else:
        target_q = reward + (1. - done) * self.gamma * np.amax(next_q, axis=1)
    # Update model
    summary, _ = self.session.run(fetches=[self.summaries, self.train],
                                  feed_dict={self.input: state,
                                              self.q_true: np.array(target_q),
                                              self.a_true: np.array(action),
                                              self.reward: np.mean(reward)})
    # Reset learn step
    self.learn_step = 0
    # Write
    self.writer.add_summary(summary, self.step)

  def replay(self, env, model_path, n_replay, plot):
    """ Model replay """
    ckpt = tf.train.latest_checkpoint(model_path)
    saver = tf.train.import_meta_graph(ckpt + '.meta')
    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name('input:0')
    output = graph.get_tensor_by_name('online/output/BiasAdd:0')
    # Replay RL agent
    state = process_frame(env.reset())
    total_reward = 0
    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        for _ in range(n_replay):
            step = 0
            while True:
                time.sleep(0.05)
                env.render()
                # Plot
                if plot:
                    if step % 100 == 0:
                        self.visualize_layer(session=sess, layer=self.conv_2, state=state, step=step)
                # Action
                if np.random.rand() <= 0:
                    action = self.actions[random.randint(0, len(self.actions) - 1)]
                else:
                    q = sess.run(fetches=output, feed_dict={input: np.expand_dims(state, 0)})
                    action = to_bin(np.argmax(q))
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = process_frame(next_state)
                step += 1
                if done:
                    break
    env.close()

  def visualize_layer(self, session, layer, state, step):
    """ Visualization auf Conv Layers"""
    units = session.run(layer, feed_dict={self.input: np.expand_dims(state, 0)})
    filters = units.shape[3]
    plt.figure(1, figsize=(40, 40))
    n_columns = 8
    n_rows = np.ceil(filters / n_columns)
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap='YlGnBu')
    plt.savefig(fname='./img/img-' + str(step) + '.png')
