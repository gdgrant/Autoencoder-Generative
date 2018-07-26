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
    'learning_rate'         : 2.5e-3,
    'task'                  : 'go',

    # Network shape
    'num_motion_dirs'       : 24,
    'num_motion_locs'       : 10,
    'n_latent'              : 12,
    'n_discriminator'       : 2,
    'n_generator'           : 100,

    'encoder_hidden'        : [200],
    'generator_hidden'      : [50, 30],
    'decoder_hidden'        : [200],
    'discriminator_hidden'  : [300],
    'solution_hidden'       : [150, 50],

    # Task setup
    'test_from_input'       : False,
    'solve_from_latent'     : False,
    'antigo'                : False,
    'nonuniform_probs'      : False,
    'pct_motion'            : 1.,
    'pct_fixation'          : 0.1,

    # Training information
    'batch_size'              : 512,
    'num_autoencoder_batches' : 8001,
    'num_GAN_batches'         : 8001,
    'num_train_batches'       : 8001,
    'num_entropy_batches'     : 2001,
    'num_final_test_batches'  : 10,

    # Costs
    'act_latent_cost'         : 16e-5,
    'gen_latent_cost'         : 16e-5,

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

    if par['test_from_input'] and par['solve_from_latent']:
        raise Exception('Only have one of [test from input] OR [solve from latent] active at a time.')

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

    sol_dim0                = par['n_latent'] if par['solve_from_latent'] else par['n_input']
    par['solution']         = [sol_dim0] + par['solution_hidden'] + [par['n_output']]


    par['discriminator_gen_target'] = np.zeros([par['batch_size'], par['n_discriminator']])
    par['discriminator_gen_target'][:,:par['n_discriminator']//2] = 1

    par['discriminator_act_target'] = np.zeros([par['batch_size'], par['n_discriminator']])
    par['discriminator_act_target'][:,par['n_discriminator']//2:] = 1

    loc_prob_set = np.random.normal(0,0.5,par['num_motion_locs'])
    dir_prob_set = np.random.normal(0,0.5,par['num_motion_dirs'])

    if par['nonuniform_probs']:
        par['locations_probs_full'] = softmax(loc_prob_set)
        par['direction_probs_full'] = softmax(dir_prob_set)

        par['locations_probs_subset'] = softmax(loc_prob_set[:par['num_motion_locs']//2])
        par['direction_probs_subset'] = softmax(dir_prob_set[:par['num_motion_dirs']//2])

    else:
        par['locations_probs_full'] = None
        par['direction_probs_full'] = None

        par['locations_probs_subset'] = None
        par['direction_probs_subset'] = None


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


update_dependencies()
print("--> Parameters successfully loaded.\n")
