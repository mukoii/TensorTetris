"""
DQN Value Agent for Tetris
Author: Nathan Delcampo
Date: 4/10/2026
Last Modified: 4/19/2026
Python Version: 3.11.14

DESC: The network is a state-value function  V(s') -> scalar
    where s' is the board feature vector AFTER a placement.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

#  Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, reward, next_state, done):
        self.buffer.append((state, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


#  Value Network
def build_value_network(feature_size: int, hidden_units: list = [128, 64, 32]) -> keras.Model:
    inputs = keras.Input(shape=(feature_size,), name="board_features")
    x = inputs

    for i, units in enumerate(hidden_units):
        x = keras.layers.Dense(units, name=f"dense_{i}")(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.ReLU()(x)

    value = keras.layers.Dense(1, activation=None, name="value")(x)
    return keras.Model(inputs=inputs, outputs=value, name="ValueNet")


#  Agent
class DQNAgent:
    """
    State/Action value-network agent.

    act(next_states)        - epsilon-greedy over afterstate values
    act_greedy(next_states) - pure greedy (demo / eval)
    """

    def __init__(
        self,

        # Agent
        feature_size: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,

        # Buffer
        batch_size: int = 512,
        buffer_size: int = 50_000,

        # Target Network
        tau: float = 0.01,

        # Agent
        hidden_units: list = [128, 64, 32],
    ):
        self.feature_size  = feature_size
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.tau           = tau

        self.replay_buffer    = ReplayBuffer(capacity=buffer_size)
        self.train_step_count = 0

        self.online_net = build_value_network(feature_size, hidden_units)
        self.target_net = build_value_network(feature_size, hidden_units)
        self._hard_update_target()

        self.optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.loss_fn   = keras.losses.Huber()

    # Action Selection
    def _best_action(self, next_states: dict, net: keras.Model) -> tuple:
        """Score every state and return the action with highest value."""
        actions = list(next_states.keys())
        features = np.array([next_states[a][0] for a in actions], dtype=np.float32)
        values = net(features, training=False).numpy().flatten()

        return actions[int(np.argmax(values))]

    def act(self, next_states: dict) -> tuple:
        if random.random() < self.epsilon:
            return random.choice(list(next_states.keys()))
        return self._best_action(next_states, self.online_net)

    def act_greedy(self, next_states: dict) -> tuple:
        return self._best_action(next_states, self.online_net)


    #Replay Buffer
    def remember(self, state_features, reward, next_state_features, done):
        self.replay_buffer.push(state_features, reward, next_state_features, done)


    def learn(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        next_values = self.target_net(next_states, training=False).numpy().flatten()
        targets = rewards + (1.0 - dones) * self.gamma * next_values

        with tf.GradientTape() as tape:
            predicted = self.online_net(states, training=True)
            predicted = tf.squeeze(predicted, axis=1)
            loss = self.loss_fn(targets, predicted)

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))

        self.train_step_count += 1
        self._soft_update_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return float(loss)

   # Target Network
    def _hard_update_target(self):
        self.target_net.set_weights(self.online_net.get_weights())

    def _soft_update_target(self):
        for ow, tw in zip(self.online_net.weights, self.target_net.weights):
            tw.assign(self.tau * ow + (1.0 - self.tau) * tw)

    #Saving and Loading
    def save(self, path: str = "tetris_dqn.weights.h5"):
        self.online_net.save_weights(path)
        print(f"[Agent] Saved: {path}")

    def load(self, path: str = "tetris_dqn.weights.h5"):
        self.online_net.load_weights(path)
        self._hard_update_target()
        print(f"[Agent] Loaded: {path}")