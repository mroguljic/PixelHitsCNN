import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


def plot_cnn_loss(history,label,img_ext):
	plt.plot(history['%s_loss'%(label)])
	plt.plot(history['val_%s_loss'%(label)])
	plt.title('%s position - model loss'%(label))
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['%s-train'%(label), '%s-validation'%(label)], loc='upper right')
	#plt.show()
	plt.savefig("plots/loss_%s_%s.png"%(label,img_ext))
	plt.close()

def plot_dnn_loss(history,label,img_ext):
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('%s position - model loss'%(label))
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['%s-train'%(label), '%s-validation'%(label)], loc='upper right')
	#plt.show()
	plt.savefig("plots/loss_%s_%s.png"%(label,img_ext))
	plt.close()


def plot_residuals(residuals,mean,sigma,RMS,label,img_ext)
	plt.hist(residuals, bins=np.arange(-60,60,0.25), histtype='step', density=True,linewidth=2, label=r'$\vartriangle$'+label)
	xmin, xmax = plt.xlim()
	x = np.linspace(xmin, xmax, 100)
	p = norm.pdf(x, mean, sigma)
	plt.title('residuals in %s, RMS = %0.2f, %s = %0.2f'%(label,RMS,r'$\sigma$',sigma))
	#plt.ylabel('No. of samples')
	plt.xlabel(r'$\mu m$')

	plt.plot(x, p, 'k', linewidth=1,color='red',label='gaussian fit')
	plt.legend()
	plt.savefig("plots/residuals_%s_%s.png"%(label,img_ext))
	plt.close()

def plot_by_clustersize(residuals,clustersize):


