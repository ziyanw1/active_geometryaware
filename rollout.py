#!/usr/bin/env python

from utils.logger import log_string
import time
from models.active_mvnet import MVInputs, SingleInput
import numpy as np
from replay_memory import trajectData

class Rollout(object):
    def __init__(self, agent, env, memory, FLAGS):
        self.agent = agent
        self.env = env
        self.mem = memory
        self.FLAGS = FLAGS
        
    def single_input_for_state(self, state):
        model_id = self.env.current_model
        
        azimuth = np.array(state[0])
        elevation = np.array(state[1])

        rgb, mask = self.mem.read_png_to_uint8(azimuth, elevation, model_id)
        invz = self.mem.read_invZ(azimuth, elevation, model_id)
        mask = (mask > 0.5).astype(np.float32) * (invz >= 1e-6)

        invz = invz[..., None]
        mask = mask[..., None]
        azimuth = azimuth[..., None]
        elevation = elevation[..., None]
        
        single_input = SingleInput(rgb, invz, mask, azimuth, elevation)
        return single_input

    def go(self, i_idx, verbose = True, add_to_mem = True):
        ''' does 1 rollout, returns mvnet_input'''

        state, model_id = self.env.reset(True)
        actions = []
        mvnet_input = MVInputs(self.FLAGS, batch_size = 1)
        
        mvnet_input.put(self.single_input_for_state(state), episode_idx = 0)
        
        for e_idx in range(1, self.FLAGS.max_episode_length):

            tic = time.time()
            agent_action = self.agent.select_action(mvnet_input, e_idx-1)
            actions.append(agent_action)
            state, next_state, done, model_id = self.env.step(actions[-1])
            
            mvnet_input.put(self.single_input_for_state(next_state), episode_idx = e_idx)

            if verbose:
                log_string('Iter: {}, e_idx: {}, azim: {}, elev: {}, model_id: {}, time: {}s'.format(
                    i_idx, e_idx, next_state[0], next_state[1], model_id, time.time()-tic
                ))
            
            if done:
                traj_state = state
                traj_state[0] += [next_state[0]]
                traj_state[1] += [next_state[1]]

                if add_to_mem:
                    temp_traj = trajectData(traj_state, actions, model_id)
                    self.mem.append(temp_traj)
                break

        return mvnet_input
