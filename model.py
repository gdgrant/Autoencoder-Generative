import numpy as np
import tensorflow as tf
import AdamOpt
from parameters import *
import go_stim as stimulus
import os, sys, time

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
print('TensorFlow version:\t', tf.__version__, '\n')

class Model:

    def __init__(self, input_data, target_data):

        self.input_data = input_data
        self.target_data = target_data

        self.initialize_variables()

        self.run_model()

        self.optimize()


    def initialize_variables(self):

        self.var_dict = {}

        for scope in par['scope_names']:
            self.var_dict[scope] = {}
            with tf.variable_scope(scope):
                for n, (s1, s2) in enumerate(zip(par[scope], par[scope][1:])):
                    self.var_dict[scope]['W{}'.format(n)] = tf.get_variable('W{}'.format(n), shape=[s1,s2])
                    self.var_dict[scope]['b{}'.format(n)] = tf.get_variable('b{}'.format(n), shape=[1,s2])

                    if (scope == 'encoder' and n == len(par['encoder'])-2) or (scope == 'generator' and n == len(par['generator'])-2):
                        self.var_dict[scope]['W{}_'.format(n)] = tf.get_variable('W{}_'.format(n), shape=[s1,s2])
                        self.var_dict[scope]['b{}_'.format(n)] = tf.get_variable('b{}_'.format(n), shape=[1,s2])


    def run_model(self):

        self.outputs_dict = {}

        for scope in ['encoder', 'generator']:
            if scope == 'encoder':
                x = self.input_data
            elif scope == 'generator':
                x = tf.random_normal(shape=[par['batch_size'],par['n_generator']])

            # Making a latent sample
            for n in range(len(par[scope][1:])):
                W = self.var_dict[scope]['W{}'.format(n)]
                b = self.var_dict[scope]['b{}'.format(n)]

                if n == len(par[scope])-2:
                    W_ = self.var_dict[scope]['W{}_'.format(n)]
                    b_ = self.var_dict[scope]['b{}_'.format(n)]

                    mu  = x @ W + b
                    sig = x @ W_ + b_

                    latent = mu + tf.exp(0.5*sig)*tf.random_normal(sig.shape)

                else:
                    x = tf.nn.relu(x @ W + b)

            self.outputs_dict[scope+'_mu']  = mu
            self.outputs_dict[scope+'_sig'] = sig
            self.outputs_dict[scope+'_lat'] = latent

            # Decoding the latent sample
            recon = latent
            for n in range(len(par['decoder'][1:])):
                W = self.var_dict['decoder']['W{}'.format(n)]
                b = self.var_dict['decoder']['b{}'.format(n)]

                if n == len(par['decoder'])-2:
                    recon = recon @ W + b
                else:
                    recon = tf.nn.relu(recon @ W + b)

            self.outputs_dict[scope+'_reconstruction'] = recon

            # Discriminating and solving from reconstrution
            for output_scope in ['discriminator', 'solution']:
                y = recon
                for n in range(len(par[output_scope][1:])):
                    W = self.var_dict[output_scope]['W{}'.format(n)]
                    b = self.var_dict[output_scope]['b{}'.format(n)]

                    if n == len(par[output_scope])-2:
                        y = y @ W + b

                        if scope == 'generator':
                            y = tf.nn.softmax(y)

                    else:
                        y = tf.nn.relu(y @ W + b)

                self.outputs_dict[scope+'_to_'+output_scope] = y


    def optimize(self):

        opt = AdamOpt.AdamOpt(tf.trainable_variables(), par['learning_rate'])
        eps = 1e-7

        # Putting together variable groups
        encoder  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        decoder  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
        VAE_vars = encoder + decoder

        generator     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        discriminator = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        GAN_vars      = generator + discriminator

        task_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='solution')

        # Task loss and training
        if par['task'] == 'trig':
            self.task_loss = tf.reduce_mean(tf.square(self.outputs_dict['encoder_to_solution']-self.target_data))
        elif par['task'] == 'go':
            self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2( \
                logits=tf.nn.relu(self.outputs_dict['encoder_to_solution']), labels=self.target_data+eps))

            y_prob = tf.nn.softmax(self.outputs_dict['generator_to_solution'])
            self.entropy_loss = -tf.reduce_mean(-y_prob * tf.log(y_prob))

            y_prob = tf.nn.softmax(self.outputs_dict['encoder_to_solution'])
            self.entropy_loss_encoded = -tf.reduce_mean(-y_prob * tf.log(y_prob))


        self.aux_loss, prev_weights, reset_prev_vars_ops = self.pathint_loss(task_vars) # Loss calculation

        self.train_task = opt.compute_gradients(self.task_loss + self.aux_loss, var_list=task_vars)
        self.train_task_entropy = opt.compute_gradients(self.entropy_loss + self.aux_loss, var_list=task_vars)

        self.pathint_stabilization(opt, prev_weights, task_vars)    # Weight stabilization
        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = opt.reset_params()


        # Autoencoder loss and training
        self.recon_loss = tf.reduce_mean(tf.square(self.outputs_dict['encoder_reconstruction']-self.input_data))

        si = self.outputs_dict['encoder_sig']
        mu = self.outputs_dict['encoder_mu']
        self.act_latent_loss = 16e-5 * -0.5*tf.reduce_mean(tf.reduce_sum(1+si-tf.square(mu)-tf.exp(si),axis=-1))

        self.train_VAE = opt.compute_gradients(self.recon_loss + self.act_latent_loss, var_list=VAE_vars)


        # Discriminator loss and training
        self.discr_gen_loss = tf.reduce_mean(tf.square(self.outputs_dict['generator_to_discriminator'] - par['discriminator_gen_target']))
        self.discr_act_loss = tf.reduce_mean(tf.square(self.outputs_dict['encoder_to_discriminator'] - par['discriminator_act_target']))

        self.gener_gen_loss = tf.reduce_mean(tf.square(self.outputs_dict['generator_to_discriminator'] - par['discriminator_act_target']))
        self.gener_act_loss = tf.reduce_mean(tf.square(self.outputs_dict['encoder_to_discriminator'] - par['discriminator_gen_target']))

        si = self.outputs_dict['generator_sig']
        mu = self.outputs_dict['generator_mu']
        self.gen_latent_loss = 16e-5 * -0.5*tf.reduce_mean(tf.reduce_sum(1+si-tf.square(mu)-tf.exp(si),axis=-1))

        self.generator_loss = self.gener_gen_loss + self.gener_act_loss + self.gen_latent_loss
        self.discriminator_loss = self.discr_gen_loss + self.discr_act_loss

        self.train_generator     = opt.compute_gradients(self.generator_loss, var_list=generator)
        self.train_discriminator = opt.compute_gradients(self.discriminator_loss, var_list=discriminator)


    def pathint_loss(self, vars):

        prev_weights = {}
        reset_prev_vars_ops = []
        self.big_omega_var = {}
        aux_losses = []

        for var in vars:
            n = var.op.name

            # Make big omegas
            self.big_omega_var[n] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            prev_weights[n] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

            # Aux loss collection
            aux_losses.append(par['omega_c']*tf.reduce_sum(self.big_omega_var[n] * (prev_weights[n]-var)**2))

            # Prev_weight resets
            reset_prev_vars_ops.append(tf.assign(prev_weights[n], var))

        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)

        return tf.add_n(aux_losses), prev_weights, reset_prev_vars_ops


    def pathint_stabilization(self, opt, prev_weights, vars):
        """ Synaptic stabilization via the Zenke method """

        # Set up method
        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  1.0)
        small_omega_var = {}

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []

        # Iterate over variables in the model
        for var in vars:
            n = var.op.name

            # Reset the small omega vars and update the big omegas
            small_omega_var[n] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            reset_small_omega_ops.append(tf.assign(small_omega_var[n], tf.zeros_like(small_omega_var[n])))
            update_big_omega_ops.append(tf.assign_add(self.big_omega_var[n], tf.div(tf.nn.relu(small_omega_var[n]), \
            	(par['omega_xi'] + tf.square(var-prev_weights[n])))))

        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        # Calculate the gradients and update the small omegas
        # This is called every batch
        with tf.control_dependencies([self.train_task]):
            self.delta_grads = opt.return_delta_grads()
            self.gradients = optimizer_task.compute_gradients(self.task_loss, var_list = vars)
            for grad,var in self.gradients:
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad))
            self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!

        # Calculate the gradients and update the small omegas
        # This is called every batch
        with tf.control_dependencies([self.train_task]):
            self.delta_grads = opt.return_delta_grads()
            self.gradients = optimizer_task.compute_gradients(self.task_loss, var_list = vars)
            for grad,var in self.gradients:
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad))
            self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!

        with tf.control_dependencies([self.train_task_entropy]):
            self.delta_grads = opt.return_delta_grads()
            self.gradients = optimizer_task.compute_gradients(self.entropy_loss, var_list = vars)
            for grad,var in self.gradients:
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad))
            self.update_small_omega_entropy = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!





def main(save_fn='testing.pkl', gpu_id=None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    stim = stimulus.GoStim()
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[par['batch_size'], par['n_input']])
    y = tf.placeholder(tf.float32, shape=[par['batch_size'], par['n_output']])

    with tf.Session() as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y)

        sess.run(tf.global_variables_initializer())
        sess.run(model.reset_prev_vars)


        print('Training Autoencoder:')
        for i in range(par['num_autoencoder_batches']):

            trial_info = stim.go_task()
            input_data  = trial_info['neural_input']
            output_data = trial_info['desired_output']

            _, recon_loss, latent_loss = sess.run([model.train_VAE, model.recon_loss, model.act_latent_loss], \
                feed_dict={x:input_data,y:output_data})

            if i%200 == 0:
                print('{:4} | Recon: {:5.3f} | Lat: {:5.3f}'.format(i, recon_loss, latent_loss))


        print('\nTraining GAN:')
        for i in range(par['num_GAN_batches']):

            for j in range(3):
                if j == 0:
                    trainer = model.train_generator
                else:
                    trainer = model.train_discriminator

                trial_info = stim.go_task()
                input_data  = trial_info['neural_input']
                output_data = trial_info['desired_output']

                _, gen_loss, discr_loss, gen_latent, outputs_dict = sess.run([trainer, model.generator_loss, \
                model.discriminator_loss, model.gen_latent_loss, model.outputs_dict], feed_dict={x:input_data,y:output_data})


                if i%200 == 0:
                    print('{:4} | {:1} | Gen: {:6.3f} | Discr: {:6.3f} | Lat: {:5.3f}'.format(i, j, gen_loss, discr_loss, gen_latent))

                """
                if i%400 == 0 and j == 2:

                    fig, ax = plt.subplots(2, 4, figsize=[12,8])

                    output = outputs_dict['generator_reconstruction'][:4]
                    output = np.reshape(output, [-1,9,10,10])

                    decisions = outputs_dict['generator_to_discriminator'][:4]

                    for k in range(4):

                        ax[0,k].imshow(np.sum(output[k], axis=0), clim=[0,3])
                        ax[0,k].set_title('Example {}, Axis 0\nD=[{:.2f}, {:.2f}]'.format(k, decisions[k][0], decisions[k][1]))
                        ax[0,k].set_axis_off()
                        ax[1,k].imshow(np.sum(output[k], axis=1), clim=[0,3])
                        ax[1,k].set_title('Example {}, Axis 1'.format(k))
                        ax[1,k].set_axis_off()

                    plt.show()#"""


        print('\nTraining Partial Task:')
        for i in range(par['num_train_batches']):

            trial_info = stim.go_task(subset=True)
            input_data  = trial_info['neural_input']
            output_data = trial_info['desired_output']

            _, _, recon_loss, task_loss, aux_loss = sess.run([model.train_task, model.update_small_omega, model.recon_loss, \
                model.task_loss, model.aux_loss], feed_dict={x:input_data,y:output_data})

            if i%200 == 0:

                # Run test
                trial_info  = stim.go_task(subset=False)
                input_data  = trial_info['neural_input']
                output_data = trial_info['desired_output']
                solutions   = sess.run(model.outputs_dict['encoder_to_solution'], feed_dict={x:input_data,y:output_data})
                acc         = np.mean(np.float32(np.equal(np.argmax(solutions, axis=1), np.argmax(output_data, axis=1))))

                print('{:4} | Recon: {:5.3f} | Task: {:5.3f} | Aux: {:5.3f} | Acc: {:5.3f}'.format( \
                    i, recon_loss, task_loss, aux_loss, acc))

        sess.run(model.update_big_omega)
        sess.run(model.reset_adam_op)
        sess.run(model.reset_prev_vars)
        sess.run(model.reset_small_omega)


        print('\nTraining Task Entropy:')
        for i in range(par['num_entropy_batches']):

            trial_info = stim.go_task(subset=False)
            input_data  = trial_info['neural_input']
            output_data = trial_info['desired_output']

            _, _, recon_loss, task_loss, entropy_loss, aux_loss = sess.run([model.train_task_entropy, model.update_small_omega_entropy, \
                model.recon_loss, model.task_loss, model.entropy_loss, model.aux_loss], feed_dict={x:input_data,y:output_data})

            if i%200 == 0:

                # Run test
                trial_info  = stim.go_task(subset=False)
                input_data  = trial_info['neural_input']
                output_data = trial_info['desired_output']
                solutions   = sess.run(model.outputs_dict['generator_to_solution'], feed_dict={x:input_data,y:output_data})
                acc         = np.mean(np.float32(np.equal(np.argmax(solutions, axis=1), np.argmax(output_data, axis=1))))

                print('{:4} | Recon: {:5.3f} | Task: {:5.3f} | Aux: {:5.3f} | Ent: {:5.3f} | Acc: {:5.3f}'.format( \
                    i, recon_loss, task_loss, aux_loss, entropy_loss, acc))

        sess.run(model.update_big_omega)
        sess.run(model.reset_adam_op)
        sess.run(model.reset_prev_vars)
        sess.run(model.reset_small_omega)

        print('\n'+'-'*80+'\nFull Task Testing:')
        acc_list = []
        for i in range(par['num_final_test_batches']):

            # Run test
            trial_info  = stim.go_task(subset=False)
            input_data  = trial_info['neural_input']
            output_data = trial_info['desired_output']
            [solutions, entropy_loss] = sess.run([model.outputs_dict['encoder_to_solution'], model.entropy_loss_encoded], feed_dict={x:input_data,y:output_data})
            acc         = np.mean(np.float32(np.equal(np.argmax(solutions, axis=1), np.argmax(output_data, axis=1))))
            acc_list.append(acc)

            print(' {} | Acc: {:5.3f}'.format(i, acc))

        recon_loss, task_loss, entropy_loss, aux_loss = sess.run([model.recon_loss, model.task_loss, model.entropy_loss, model.aux_loss], feed_dict={x:input_data,y:output_data})

        print('Test | Recon: {:5.3f} | Task: {:5.3f} | Aux: {:5.3f} | Mean Acc: {:5.3f}'.format( \
            i, recon_loss, task_loss, aux_loss, np.mean(acc_list)))
        print('-'*80)

if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main('testing.pkl', sys.argv[1])
        else:
            main('testing.pkl')
    except KeyboardInterrupt:
        print('Quit by KeyboardInterrupt.')
