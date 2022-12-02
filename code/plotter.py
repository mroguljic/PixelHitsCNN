import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import optimize
import ROOT
from ROOT import *

gStyle.SetOptStat(1)
gROOT.SetBatch(1)
gStyle.SetOptFit(1);

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5*((x - mean) / stddev)**2)

def plot_cnn_loss(history,label,img_ext):
	plt.plot(history['%s_loss'%(label)])
	plt.plot(history['val_%s_loss'%(label)])
	plt.title('%s position - model loss'%(label))
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['%s-train'%(label), '%s-validation'%(label)], loc='upper right')
	#plt.show()
	plt.savefig("plots/python/loss/loss_%s_%s.png"%(label,img_ext))
	plt.close()

def plot_dnn_loss(history,label,img_ext):
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('%s position - model loss'%(label))
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['%s-train'%(label), '%s-validation'%(label)], loc='upper right')
	#plt.show()
	plt.savefig("plots/python/loss/loss_%s_%s.png"%(label,img_ext))
	plt.close()


def plot_residuals(residuals,algo,label,img_ext):

	res = ROOT.TH1F("residuals","%s %s"%(algo,label),50,-20,20)
	res.Sumw2() # statistical uncertainties to be calculated using the sum of weights squared
	'''
	Once the histogram has been filled, we want to make sure that it doesnt disappear. By default, histograms
	are linked to the last file that was opened, so when you close the file the histogram may disappear.
	We want to change this behaviour, to say that the histogram we just created has no parent file, and thus
	should not be removed when any files are closed.
	'''
	res.SetDirectory(0)
	for entry in residuals:
		res.Fill(entry)
	print("RMS of %s %s = %f"%(algo,label,res.GetRMS()))
	canvas = ROOT.TCanvas (" canvas ")
	canvas.cd()
	res.SetMarkerColor(ROOT.kRed);
    	res.SetMarkerStyle(20)
	res.SetMarkerSize(0.6)
	res.SetLineColor(ROOT.kRed)
	res.GetXaxis().SetTitle(r'$\mu m$')
	res.GetYaxis().SetTitle("Number of events")
	res.SetTitle("%s - residuals in %s"%(algo, label))
	res.Fit("gaus","E")
	res.GetFunction("gaus").SetLineColor(ROOT.kBlack);
	res.Draw("pe")
	canvas.Print("plots/python/residuals/%s_residuals_%s.png"%(label,img_ext))

def plot_cot(cot_list,label,img_ext):
	cot_hist = ROOT.TH1F(label,label, 20, -0.4,0.4)
	cot_hist.Sumw2()
	cot_hist.SetDirectory(0)
	for entry in cot_list:
		cot_hist.Fill(entry)
	canvas = ROOT.TCanvas("canvas")
	canvas.cd()
	cot_hist.SetTitle("Distribution of cot #alpha")
	cot_hist.SetLineWidth(2)
	cot_hist.Draw("hist")
	canvas.Print("plots/python/%s_dist_%s.png"%(label,img_ext))
def plot_by_clustersize(residuals,clustersize,label,img_ext):

	residuals = np.asarray(residuals)
	clustersize = np.asarray(clustersize)
	max_size = int(np.amax(clustersize))
	sigma_per_size = np.zeros((max_size+1,1))
	print(residuals.shape)
	pp = PdfPages('plots/res_%s_csize_%s.pdf'%(label,img_ext))

	for i in range(1,max_size+1):
		indices = np.argwhere(clustersize==i)[:,0]
		print(i,':',indices.shape)
		if(len(indices)==0):
			sigma_per_size[i]=0.
		else:
			residuals_per_size = residuals[indices]
			sigma_per_size[i] = np.std(residuals_per_size)
			plt.hist(residuals_per_size, bins=np.arange(-300,300,0.25), histtype='step',linewidth=2, label=r'$\vartriangle$'+label)
			plt.xlabel(r'$\mu m$')
			plt.title('residuals in %s for clustersize = %i, %s = %0.2f'%(label,i,r'$\sigma$',sigma_per_size[i]))
			pp.savefig()
			plt.close()

	pp.close()

	x = np.linspace(1, max_size, max_size)
	print(np.amin(clustersize),max_size)
	print(x.shape,sigma_per_size[1:].shape)
	plt.scatter(x,sigma_per_size[1:])
	plt.xlabel('clustersize in %s'%(label))
	plt.ylabel('resolution in %s'%(label))
	plt.title('resolution vs clustersize in %s'%(label))
	plt.savefig("plots/python/per_clustersize/resvssize_%s_%s"%(label,img_ext))
	plt.close()

