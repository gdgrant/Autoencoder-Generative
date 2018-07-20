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

        scope = 'solution'
        self.var_dict[scope] = {}
        with tf.variable_scope(scope):
            for n, (s1, s2) in enumerate(zip(par[scope], par[scope][1:])):
                self.var_dict[scope]['W{}'.format(n)] = tf.get_variable('W{}'.format(n), shape=[s1,s2])
                self.var_dict[scope]['b{}'.format(n)] = tf.get_variable('b{}'.format(n), shape=[1,s2])


    def run_model(self):

        scope = 'solution'
        y = self.input_data
        for n in range(len(par[scope][1:])):
            W = self.var_dict[scope]['W{}'.format(n)]
            b = self.var_dict[scope]['b{}'.format(n)]

            y = tf.nn.relu(y @ W + b)

        self.y_hat = y

    def optimize(self):

        opt = AdamOpt.AdamOpt(tf.trainable_variables(), par['learning_rate'])
        eps = 1e-7

        # Task loss and training
        if par['task'] == 'trig':
            self.task_loss = tf.reduce_mean(tf.square(self.outputs_dict['encoder_to_solution']-self.target_data))
        elif par['task'] == 'go':
            self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2( \
                logits=self.y_hat, labels=self.target_data+eps))

            y_prob = tf.nn.softmax(self.y_hat)
            self.entropy_loss = -tf.reduce_mean(-y_prob * tf.log(y_prob))

        self.train_task = opt.compute_gradients(self.task_loss)
        self.train_task_entropy = opt.compute_gradients(self.entropy_loss)


def main(gpu_id=None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    print_variables()

    stim = stimulus.GoStim()
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[par['batch_size'], par['n_input']])
    y = tf.placeholder(tf.float32, shape=[par['batch_size'], par['n_output']])

    with tf.Session() as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y)

        sess.run(tf.global_variables_initializer())

        print('Training Partial:')
        for i in range(par['num_train_batches']):

            trial_info = stim.go_task(subset=True)
            input_data  = trial_info['neural_input']
            output_data = trial_info['desired_output']

            _, solutions, task_loss = sess.run([model.train_task, model.y_hat, model.task_loss], feed_dict={x:input_data,y:output_data})
            acc = np.mean(np.float32(np.equal(np.argmax(solutions, axis=1), np.argmax(output_data, axis=1))))

            if i%200 == 0:
                print('{:4} | Task: {:5.3f} | Partial Acc: {:5.3f}'.format(i, task_loss, acc))

        """
        print('\nTraining Entropy:')
        for i in range(par['num_entropy_batches']):

            trial_info = stim.go_task(subset=False)
            input_data  = trial_info['neural_input']
            output_data = trial_info['desired_output']

            _, solutions, task_loss, entropy_loss = sess.run([model.train_task_entropy, model.y_hat, model.task_loss, model.entropy_loss], feed_dict={x:input_data,y:output_data})
            acc = np.mean(np.float32(np.equal(np.argmax(solutions, axis=1), np.argmax(output_data, axis=1))))

            if i%200 == 0:
                print('{:4} | Task: {:5.3f} | Entropy: {:5.3} | '.format(i, task_loss, entropy_loss))
            """



        print('\nTesting Full:')
        for i in range(par['num_final_test_batches']):

            # Run test
            trial_info  = stim.go_task(subset=False)
            input_data  = trial_info['neural_input']
            output_data = trial_info['desired_output']
            [solutions, entropy_loss] = sess.run([model.y_hat, model.entropy_loss], feed_dict={x:input_data,y:output_data})
            acc         = np.mean(np.float32(np.equal(np.argmax(solutions, axis=1), np.argmax(output_data, axis=1))))
            print('{:4} | Task: {:5.3f} | Full Acc: {:5.3f}'.format(i, task_loss, acc))




def print_variables():

    checked_keys = ['learning_rate', 'num_motion_dirs', 'num_motion_locs', 'n_latent', \
        'num_autoencoder_batches', 'num_GAN_batches', 'num_train_batches', 'num_entropy_batches', \
        'act_latent_cost', 'gen_latent_cost', 'test_from_input']

    print('')
    for k in checked_keys:
        print(k.ljust(25), '|', par[k])
    print('')


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main(sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        print('Quit by KeyboardInterrupt.')
