import numpy as np
import matplotlib.pyplot as plt
from parameters import *

class GoStim:

    def __init__(self):

        self.input_shape    = [par['batch_size'], par['n_input']]
        self.output_shape   = [par['batch_size'], par['n_output']]

        self.motion_dirs    = np.linspace(0,2*np.pi-2*np.pi/par['num_motion_dirs'], par['num_motion_dirs'])
        self.trial_info     = {}


    def go_task(self, subset=False):

        # Setting up
        
        resp = np.zeros([par['batch_size'], par['num_motion_dirs']+1, par['num_motion_locs'], par['num_motion_locs']])

        pm = [1., 0.]       # [0.75, 0.25]
        pf = [0., 1.]       # [0.25, 0.75]

        motion   = np.random.choice([True, False], size=par['batch_size'], p=pm)
        fixation = np.random.choice([True, False], size=par['batch_size'], p=pf)

        locs = par['num_motion_locs']//2 if subset else par['num_motion_locs']
        locations = np.random.choice(locs, size=[par['batch_size'],2])
        direction = np.random.choice(par['num_motion_dirs'], size=[par['batch_size'],1])

        x_ref = np.arange(par['num_motion_locs'])[:,np.newaxis] * np.ones([1,par['num_motion_locs']])
        y_ref = np.transpose(x_ref)

        # Calculating

        x_locational = locations[:,0,np.newaxis,np.newaxis] - x_ref[np.newaxis,...]
        y_locational = locations[:,1,np.newaxis,np.newaxis] - y_ref[np.newaxis,...]

        spatial = np.exp(-1/2 * (np.square(x_locational)+np.square(y_locational)))
        angular = np.angle(np.exp(1j*self.motion_dirs[direction] - 1j*self.motion_dirs[np.newaxis,...]))
        stim_motion = np.exp(-1/2 * np.square(angular))

        activity  = spatial[...,np.newaxis] * stim_motion[:,np.newaxis,np.newaxis,:]
        activity *= (1. + np.random.gamma(1., size=[par['batch_size'],1,1,1]))

        motion_exp = np.reshape(motion, [-1,1,1,1])
        resp[:,:-1,:,:]     = np.where(motion_exp, np.transpose(activity, [0,3,1,2]), np.zeros([1,1,1,1]))
        resp[:,-1,3:7,3:7]  = np.where(fixation, 1., 0.)[:,np.newaxis,np.newaxis]

        # x-loc, y-loc, direction, motion on/off, fixation on/off
        self.trial_info['input'] = np.concatenate([locations, direction, motion[:,np.newaxis], fixation[:,np.newaxis]], axis=1)

        # Neural input plus noise
        self.trial_info['neural_input']   = np.reshape(resp, [par['batch_size'], -1]) + np.random.normal(0, 0.1, size=self.input_shape)

        if par['task'] == 'trig':
            # Desired output, multiplied by opposite fixation
            self.trial_info['desired_output'] = np.concatenate([np.cos(self.motion_dirs[direction]), \
                                                    np.sin(self.motion_dirs[direction])], axis=1) * np.where(fixation, 0., 1.)[:,np.newaxis]
        elif par['task'] == 'go':
            # Labels corresponding to direction
            output = np.zeros(self.output_shape)
            for i, (d, f) in enumerate(zip(np.squeeze(direction), fixation)):
                output[i,d] = 0 if f else 1
            self.trial_info['desired_output'] = output

        return self.trial_info
