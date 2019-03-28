### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time
from itertools import product

# Plotting suite
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model modules
from parameters import *
import stimulus

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Model:

	def __init__(self, input_data, target_data, keep_prob):

		self.input_data = input_data
		self.target_data = tf.one_hot(target_data, par['n_output'])
		self.keep_prob = keep_prob

		self.var_dict = {}
		self.run_model()
		self.optimize()


	def run_model(self):

		x = self.input_data

		# Convolutional layers to latent
		print('In   :', x.shape)
		print('-'*10)
		conv_shapes = [x.shape.as_list()]
		for i in range(par['num_conv_layers']):
			self.var_dict[f'conv_k{i}'] = tf.get_variable(f'conv_k{i}', initializer=tf.random_uniform([5,5,1,1], -0.1, 0.1))
			x = tf.nn.conv2d(x, self.var_dict[f'conv_k{i}'], strides=[1,2,2,1], padding='SAME')
			print('Conv :', x.shape)
			conv_shapes.append(x.shape.as_list())

		# Dense layers to output
		h = tf.reshape(x, [par['batch_size'],-1])
		print('-'*10)
		print('Hid  :', h.shape)
		for i, nh in enumerate(par['n_hidden']):
			self.var_dict[f'W_hid{i}'] = tf.get_variable(f'W_hid{i}', initializer=tf.random_uniform([h.shape.as_list()[1], nh], -0.1, 0.1))
			self.var_dict[f'b_hid{i}'] = tf.get_variable(f'b_hid{i}', initializer=tf.zeros([1, nh]))

			h = tf.nn.dropout(tf.nn.relu(h @ self.var_dict[f'W_hid{i}'] + self.var_dict[f'b_hid{i}']), keep_prob=self.keep_prob)
			print('Hid  :', h.shape)

		self.var_dict['W_out'] = tf.get_variable('W_out', initializer=tf.random_uniform([h.shape.as_list()[1], par['n_output']], -0.1, 0.1))
		self.var_dict['b_out'] = tf.get_variable('b_out', initializer=tf.zeros([1, par['n_output']]))
		self.output = h @ self.var_dict['W_out'] + self.var_dict['b_out']

		print('Out  :', self.output.shape)

		# Deconvolutional layers to reconstruction
		self.latent = x
		print('-'*10)
		print('Lat  :', x.shape)
		for i, sh in zip(range(par['num_conv_layers']), conv_shapes[-2::-1]):
			self.var_dict[f'deconv_k{i}'] = tf.get_variable(f'deconv_k{i}', initializer=tf.random_uniform([5,5,1,1], -0.1, 0.1))
			x = tf.nn.conv2d_transpose(x, self.var_dict[f'deconv_k{i}'], sh, strides=[1,2,2,1], padding='SAME')
			print('Dec  :', x.shape)
		self.recon = tf.nn.relu(x)


	def optimize(self):

		opt = tf.train.AdamOptimizer()

		self.task_loss = par['task_cost'] * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(\
			labels=self.target_data, logits=self.output))

		self.recon_loss = par['recon_cost'] * tf.reduce_mean(tf.square(self.input_data - self.recon))

		self.latent_loss = par['latent_cost'] * tf.reduce_mean(tf.square(self.latent))

		total_loss = self.task_loss + self.recon_loss + self.latent_loss
		self.train = opt.minimize(total_loss)




def main(gpu_id = None):

	# Print out context
	# Select GPU
	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	# Reduce memory consumption for GPU 0
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) \
		if gpu_id == '3' else tf.GPUOptions()

	# Initialize stimulus environment
	stim = stimulus.Stimulus()

	# Reset graph and designate placeholders
	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['batch_size'], 28, 28, 1], 'input')
	y = tf.placeholder(tf.int32, [par['batch_size']], 'output')
	k = tf.placeholder(tf.float32, [], 'keep_prob')

	# Start TensorFlow session
	with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:

		# Set up and initialize model on desired device
		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, y, k)
		sess.run(tf.global_variables_initializer())

		# Start training loop
		print('-'*20 + '\nStarting training.')
		for i in range(par['iterations']):

			images, labels = stim.make_batch()
			train_get_list = [model.train]
			_, = sess.run(train_get_list, feed_dict={x:images,y:labels,k:par['dropout_keep_prob']})

			if i%100 == 0:

				images, labels = stim.make_batch(test=True)
				test_get_list = [model.output, model.recon, model.task_loss, model.recon_loss, model.latent_loss]
				output, recon, task_loss, recon_loss, latent_loss = sess.run(test_get_list, feed_dict={x:images,y:labels,k:1.})

				acc = np.mean(np.argmax(output, axis=-1) == labels)
				print('Iter {:>5} | Accuracy: {:5.3f} | Task Loss: {:5.3f} | Recon Loss: {:5.3f} | Latent Loss: {:5.3f}'.format(\
					i, acc, task_loss, recon_loss, latent_loss))

		print('\nTraining complete.  Recording results.')

		images, labels = stim.make_batch(test=True)
		test_get_list = [model.output, model.recon, model.task_loss, model.recon_loss, model.latent_loss]
		output, recon, task_loss, recon_loss, latent_loss = sess.run(test_get_list, feed_dict={x:images,y:labels,k:1.})
		acc = np.mean(np.argmax(output, axis=-1) == labels)

		data = {
			'parameters'	: par,
			'images'		: images,
			'labels'		: labels,
			'recons'		: recon,
			'weights'		: sess.run([model.var_dict])[0],
			'accuracy'		: acc
		}

		pickle.dump(data, open('./savedir/' + par['savefn'] + '.pkl', 'wb'))
		print('Results saved.  Model complete.')


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')