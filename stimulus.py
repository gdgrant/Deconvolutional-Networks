from parameters import *
import numpy as np
import pickle


class Stimulus:

	def __init__(self):

		self.mnist_dir = './mnist/data/original/'
		self.generate_mnist_tuning()

	def generate_mnist_tuning(self):

		# Import MNIST data
		from mnist import MNIST
		mndata = MNIST(self.mnist_dir)
		self.train_images, self.train_labels = mndata.load_training()
		self.test_images, self.test_labels = mndata.load_testing()

		# Get number of training and testing examples
		self.num_training = len(self.train_images)
		self.num_testing = len(self.test_images)

		# Convert to arrays
		self.train_images = np.array(self.train_images).astype(np.float32)/255
		self.test_images = np.array(self.test_images).astype(np.float32)/255

		self.train_labels = np.array(self.train_labels).astype(np.int32)
		self.test_labels = np.array(self.test_labels).astype(np.int32)

	def make_batch(self, test=False):

		ind_num = self.num_testing if test else self.num_training
		images  = self.test_images if test else self.train_images
		labels  = self.test_labels if test else self.train_labels
		
		q = np.random.choice(ind_num, size=par['batch_size'])
		batch_images = images[q,:].reshape([par['batch_size'],28,28,1])
		batch_labels = labels[q]

		return batch_images, batch_labels