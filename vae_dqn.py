## Copyright (C) 2016-17 Google Inc.
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License along
## with this program; if not, write to the Free Software Foundation, Inc.,
## 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
################################################################################
"""A working example of deepmind_lab using python."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import six
import cv2

from jaco_arm import JacoEnv

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Reshape, Permute, Activation, Conv3D, Lambda, Input, Concatenate
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomUniform
import keras.backend as K
from keras.callbacks import ModelCheckpoint


from keras.utils import multi_gpu_model

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from factored_variational_autoencoder_deconv import vae

WEIGHTS_FILE = "jaco_myvae_weights.h5"

HIDDEN_SIZE = 256
NUM_HIDDEN_LAYERS = 5
WINDOW_LENGTH = 4

MULTI_GPU = False

def run():
    """Construct and start the environment."""

    env = JacoEnv(64,
                  64,
                  100,
                  0.1,
                  0.8,
                  True)
    nb_actions = env.real_num_actions
    new_floor_color = list((0.55 - 0.45) * np.random.random(3) + 0.45) + [1.]
    new_cube_color = list(np.random.random(3)) + [1.]
    env.change_floor_color(new_floor_color)
    env.change_cube_color(new_cube_color)

    global vae
    vae.load_weights(WEIGHTS_FILE)
    print("#########################")
    nb_observation_space = (64, 64, 3)
    original_input = Input(shape=(WINDOW_LENGTH,) + nb_observation_space)
    in_layer = [Lambda(lambda x: x[:, i, :, :])(original_input) for i in range(WINDOW_LENGTH)]
    s_output_layer = Lambda(lambda x: x[:, 32:])(vae.layers[-2].outputs[2])
    vae = Model(vae.inputs, [s_output_layer])
    for layer in vae.layers:
        layer.trainable = False
    print(vae.summary())
    vae_output = [vae(x) for x in in_layer]

    x = Concatenate()(vae_output)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(nb_actions, activation='linear')(x)
    model = Model(original_input, [x])
    print(model.summary())
    if MULTI_GPU:
        model = multi_gpu_model(model, gpus=2)
        print(model.summary())

    num_warmup = 50000
    num_simulated_annealing = 500000 + num_warmup

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=num_simulated_annealing)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, nb_steps_warmup=num_warmup, gamma=.99, target_model_update=10000, train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    if False:
        checkpoint_callback = ModelCheckpoint("myvae_dqn_checkpoint", monitor='episode_reward', verbose=0, save_best_only=True, save_weights_only=True, mode='max', period = 10)
        history = dqn.fit(env, nb_steps=num_simulated_annealing + 450000, visualize=False, verbose=1, callbacks=[checkpoint_callback])
        dqn.save_weights("myvae_dqn_weights")
        np.savez_compressed("myvae_dqn_history", episode_reward=np.asarray(history.history['episode_reward']))
    else:
        dqn.load_weights("myvae_dqn_weights")

        print("original domain")
        source_test_losses = dqn.test(env, nb_episodes=100, visualize=True)
        np.savez_compressed("myvae_dqn_source_test",
                            episode_reward=np.asarray(source_test_losses.history['episode_reward']),
                            nb_steps=np.asarray(source_test_losses.history['nb_steps']))
        source_as_np = np.array(source_test_losses.history['episode_reward'])

        print("target domain")
        new_floor_color = [0.4, 0.6, 0.4, 1.]
        new_cube_color = [1.0, 0.0, 0.0, 1.]
        env.change_floor_color(new_floor_color)
        env.change_cube_color(new_cube_color)
        target_test_losses = dqn.test(env, nb_episodes=100, visualize=True)
        np.savez_compressed("myvae_dqn_target_test",
                            episode_reward=np.asarray(target_test_losses.history['episode_reward']),
                            nb_steps=np.asarray(target_test_losses.history['nb_steps']))
        target_as_np = np.array(target_test_losses.history['episode_reward'])
        print(source_as_np.min(), source_as_np.mean(), source_as_np.max())
        print(target_as_np.min(), target_as_np.mean(), target_as_np.max())

if __name__ == '__main__':
    run()
