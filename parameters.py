import numpy as np


par = {
	'savefn'				: 'task_and_recon',
	'batch_size'			: 512,
	'iterations'			: 10000,
	'n_hidden'				: [200,200],
	'n_output'				: 10,

	'num_conv_layers'		: 2,
	'dropout_keep_prob'		: 0.75,

	'task_cost'				: 1.,
	'recon_cost'			: 1.,
	'latent_cost'			: 1e-3,
	'learning_rate'			: 5e-4,
}