### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
import tensorflow as tf
from itertools import product, chain
import matplotlib.pyplot as plt

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par

par = {
    # General parameters
    'save_dir'              : './savedir/',
    'learning_rate'         : 5e-3,
    'task'                  : 'go',

    # Network shape
    'num_motion_dirs'       : 8,
    'num_motion_locs'       : 10,
    'n_latent'              : 12,
    'n_discriminator'       : 2,
    'n_generator'           : 100,

    'encoder_hidden'        : [200],
    'generator_hidden'      : [50, 30],
    'decoder_hidden'        : [200],
    'discriminator_hidden'  : [300],
    'solution_hidden'       : [150, 50],

    # Training information
    'batch_size'              : 512,
    'num_autoencoder_batches' : 2001,
    'num_GAN_batches'         : 2001,
    'num_train_batches'       : 2001,
    'num_entropy_batches'     : 2001,
    'num_final_test_batches'  : 10,

    # Stabilization
    'omega_c'                 : 0.0,
    'omega_xi'                : 0.01,
}

############################
### Dependent parameters ###
############################

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for (key, val) in updates.items():
        par[key] = val
        print('Updating : ', key, ' -> ', val)
    update_dependencies()


def update_dependencies():

    par['n_input']  = par['num_motion_locs']**2 * (par['num_motion_dirs']+1)

    if par['task'] == 'trig':
        par['n_output'] = 2
    elif par['task'] == 'go':
        par['n_output'] = par['num_motion_dirs'] + 1

    par['scope_names'] = ['encoder', 'decoder', 'generator', 'discriminator', 'solution']

    par['encoder']          = [par['n_input']] + par['encoder_hidden'] + [par['n_latent']]
    par['generator']        = [par['n_generator']] + par['generator_hidden'] + [par['n_latent']]
    par['decoder']          = [par['n_latent']] + par['decoder_hidden'] + [par['n_input']]
    par['discriminator']    = [par['n_input']] + par['discriminator_hidden'] + [2]
    par['solution']         = [par['n_input']] + par['solution_hidden'] + [par['n_output']]


    par['discriminator_gen_target'] = np.zeros([par['batch_size'], par['n_discriminator']])
    par['discriminator_gen_target'][:,:par['n_discriminator']//2] = 1

    par['discriminator_act_target'] = np.zeros([par['batch_size'], par['n_discriminator']])
    par['discriminator_act_target'][:,par['n_discriminator']//2:] = 1

update_dependencies()
print("--> Parameters successfully loaded.\n")
