import numpy as np
import pickle
import matplotlib.pyplot as plt



data = pickle.load(open('./savedir/task_and_recon.pkl', 'rb'))
images = np.squeeze(data['images'])
recons = np.squeeze(data['recons'])


def show_recon_examples():
	trials = 8
	cmax = np.maximum(images.max(), recons.max())
	fig, ax = plt.subplots(2,trials,figsize=[16,6])
	for i in range(trials):
		ind = i
		ax[0,i].imshow(images[ind,:,:], aspect='auto', clim=(0,cmax))
		ax[1,i].imshow(recons[ind,:,:], aspect='auto', clim=(0,cmax))

		ax[0,i].set_title(f'Image {ind}')
		ax[1,i].set_title(f'Recon {ind}')

		loss = np.mean(np.square(images[ind,:,:] - recons[ind,:,:]))
		ax[1,i].set_xlabel('$\\ell={:5.3f}$'.format(loss))

		ax[0,i].set_xticks([])
		ax[0,i].set_yticks([])
		ax[1,i].set_xticks([])
		ax[1,i].set_yticks([])

	plt.tight_layout()
	plt.show()


show_recon_examples()