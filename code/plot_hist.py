import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import optimize
import ROOT
from ROOT import *

img_ext = '103121_2024'
SIMHITPERCLMAX = 10

gStyle.SetOptStat(1)
gROOT.SetBatch(1)
gStyle.SetOptFit(1);

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5*((x - mean) / stddev)**2)


def plot_residual(residuals,label,algo):
	
	bins = np.linspace(-300,300,100)
	residuals*=1e4

	print("====== %s %s ======"%(algo,label))

	#residuals = residuals[residuals<1000]
	'''
	residuals = np.zeros_like(results)+99999.

	for i in range(len(results)):
		for j in range(SIMHITPERCLMAX):
			if(abs(results[i]-sim[i][j])<residuals[i]):
				#if i==16049: 
				#	print("idx 16049: ",results[i], sim[i][j],abs(results[i]-sim[i][j]),residuals[i])
				residuals[i] = (results[i]-sim[i][j])

			#else: print("abs(results[i]-sim[i][j])<residuals[i]",results[i])
	
	#residuals = residuals[residuals<1000]
	'''

	#print("no of residuals >1000: ",len(np.argwhere(residuals>1000)))
	#print(np.argwhere(residuals>1000),residuals[residuals>1000])
	RMS = np.std(residuals)

	binned_data,bins_h,patches = plt.hist(residuals, bins=bins, histtype='step', density=False,linewidth=2, alpha=0.)
	
	bins_g = np.zeros_like(bins)

	for i in range(len(bins)-1):
		bins_g[i] = (bins[i]+bins[i+1])/2.
	
	bins_g = bins[:-1]
	
	popt, _ = optimize.curve_fit(gaussian, bins_g, binned_data)

	plt.scatter(bins_g,binned_data,marker='o',s=10,label=r'$\vartriangle$'+label)
	plt.plot(bins_g, gaussian(bins_g, *popt),linewidth=1,color='black',label='gaussian fit')

	print("popt = ",popt)
	plt.title('%s - residuals in %s, RMS = %0.2f, %s = %0.2f'%(algo, label,RMS,r'$\sigma$',popt[2]))
	#plt.ylabel('No. of samples')
	plt.xlabel(r'$\mu m$')
	plt.legend()
	plt.savefig("plots/CMSSW/residuals/%s_residuals_%s_%s.png"%(label,algo,img_ext))
	plt.close()

def plot_root(residuals,label,algo):

	res = ROOT.TH1F("residuals","%s %s"%(algo,label),100,-300,300)
	res.Sumw2() # statistical uncertainties to be calculated using the sum of weights squared
	'''
	Once the histogram has been filled, we want to make sure that it doesnt disappear. By default, histograms
	are linked to the last file that was opened, so when you close the file the histogram may disappear.
	We want to change this behaviour, to say that the histogram we just created has no parent file, and thus
	should not be removed when any files are closed.
	'''
	residuals*=1e4
	res.SetDirectory(0)
	for entry in residuals:
		res.Fill(entry)

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
	canvas.Print("plots/CMSSW/residuals/%s_residuals_%s_%s.png"%(label,algo,img_ext))



def plot_by_clustersize(residuals_x,residuals_y,algo,img_ext):
	#print clustersize wise residuals
	bins = np.linspace(-300,300,100)

	for label in ['x','y']:

		pp = PdfPages('plots/CMSSW/per_clustersize/res_vs_csize_%s_%s_%s.pdf'%(algo,label,img_ext))
		clustersize_res = np.genfromtxt("txt_files/cnn2d_MC_perclustersize_%s.txt"%label)
		if algo=='1dcnn': clustersize_res = np.genfromtxt("txt_files/cnn1d_MC_perclustersize_%s.txt"%label)

		cl1 = clustersize_res[:,0]
		cl2 = clustersize_res[:,1]
		cl3 = clustersize_res[:,2]
		cl4 = clustersize_res[:,3]
		cl5 = clustersize_res[:,4]
		cl6 = clustersize_res[:,5]

		if label=='x': residuals = residuals_x
		else: residuals = residuals_y 

		
		cl1 = residuals[cl1!=-999.]
		cl2 = residuals[cl2!=-999.]
		cl3 = residuals[cl3!=-999.]
		cl4 = residuals[cl4!=-999.]
		cl5 = residuals[cl5!=-999.]
		cl6 = residuals[cl6!=-999.]



		plt.hist(cl1, bins=bins, histtype='step', linewidth=2)
		plt.title('%s - residuals in %s, clustersize = 1'%(algo,label))
		pp.savefig()
		plt.close()

		plt.hist(cl2, bins=bins, histtype='step',linewidth=2)
		plt.title('%s - residuals in %s, clustersize = 2'%(algo,label))
		pp.savefig()
		plt.close()

		plt.hist(cl3, bins=bins, histtype='step', linewidth=2)
		plt.title('%s - residuals in %s, clustersize = 3'%(algo,label))
		pp.savefig()
		plt.close()

		plt.hist(cl4, bins=bins, histtype='step', linewidth=2)
		plt.title('%s - residuals in %s, clustersize = 4'%(algo,label))
		pp.savefig()
		plt.close()

		plt.hist(cl5, bins=bins, histtype='step', linewidth=2)
		plt.title('%s - residuals in %s, clustersize = 5'%(algo,label))
		pp.savefig()
		plt.close()

		plt.hist(cl6, bins=bins, histtype='step', linewidth=2)
		plt.title('%s - residuals in %s, clustersize = 6'%(algo,label))
		pp.savefig()
		plt.close()

		pp.close()

cnn1d_x = np.genfromtxt("txt_files/cnn1d_MC_x.txt")[:,1]
cnn1d_y = np.genfromtxt("txt_files/cnn1d_MC_y.txt")[:,1]
cnn1d_ids = np.genfromtxt("txt_files/cnn1d_MC_y.txt")[:,0]

cnn1d_x_det = np.genfromtxt("txt_files/cnn1d_MC_x_detangles.txt")[:,1]
cnn1d_y_det = np.genfromtxt("txt_files/cnn1d_MC_y_detangles.txt")[:,1]

cnn2d_x = np.genfromtxt("txt_files/cnn2d_MC_x.txt")[:,1]
cnn2d_y = np.genfromtxt("txt_files/cnn2d_MC_y.txt")[:,1]
cnn2d_ids = np.genfromtxt("txt_files/cnn2d_MC_y.txt")[:,0]

cnn2d_x_det = np.genfromtxt("txt_files/cnn2d_MC_x_detangles.txt")[:,1]
cnn2d_y_det = np.genfromtxt("txt_files/cnn2d_MC_y_detangles.txt")[:,1]

gen_x = np.genfromtxt("txt_files/generic_MC_x.txt")[:,1]
gen_y = np.genfromtxt("txt_files/generic_MC_y.txt")[:,1]
gen_ids = np.genfromtxt("txt_files/generic_MC_y.txt")[:,0]

gen_x_det = np.genfromtxt("txt_files/generic_MC_x_detangles.txt")[:,1]
gen_y_det = np.genfromtxt("txt_files/generic_MC_y_detangles.txt")[:,1]

template_x = np.genfromtxt("txt_files/template_MC_x.txt")[:,1]
template_y = np.genfromtxt("txt_files/template_MC_y.txt")[:,1]
template_ids = np.genfromtxt("txt_files/template_MC_y.txt")[:,0]

print("gen_x shape = ",gen_x.shape,"gen_y shape = ",gen_y.shape)
print("template_x shape = ",template_x.shape,"template_y shape = ",template_y.shape)
print("cnn1d_x shape = ",cnn1d_x.shape,"cnn1d_y shape = ",cnn1d_y.shape)
print("cnn2d_x shape = ",cnn2d_x.shape,"cnn2d_y shape = ",cnn2d_y.shape)
print("gen_x_det shape = ",gen_x_det.shape,"gen_y_det shape = ",gen_y_det.shape)
print("cnn1d_x_det shape = ",cnn1d_x_det.shape,"cnn1d_y_det shape = ",cnn1d_y_det.shape)
print("cnn2d_x_det shape = ",cnn2d_x_det.shape,"cnn2d_y_det shape = ",cnn2d_y_det.shape)

n = min(len(cnn1d_ids),len(template_ids))
print(len(cnn1d_ids[:n]==template_ids[:n]))

residuals_x = plot_root(cnn1d_x,'x','1dcnn')
residuals_y = plot_root(cnn1d_y,'y','1dcnn')

#residuals_x = plot_root(cnn1d_x_det,'x','1dcnn_detangles')
#residuals_y = plot_root(cnn1d_y_det,'y','1dcnn_detangles')
#plot_by_clustersize(residuals_x,residuals_y,'1dcnn',img_ext)

#residuals_x = plot_residual(dnn_x,simhits_x,'x','dnn')
#residuals_y = plot_residual(dnn_y,simhits_y,'y','dnn')

residuals_x = plot_root(gen_x,'x','gen')
residuals_y = plot_root(gen_y,'y','gen')

#residuals_x = plot_root(gen_x_det,'x','gen_detangles')
#residuals_y = plot_root(gen_y_det,'y','gen_detangles')

residuals_x = plot_root(template_x,'x','template')
residuals_y = plot_root(template_y,'y','template')
'''
residuals_x = plot_root(cnn2d_x,'x','2dcnn')
residuals_y = plot_root(cnn2d_y,'y','2dcnn')

residuals_x = plot_root(cnn2d_x_det,'x','2dcnn_detangles')
residuals_y = plot_root(cnn2d_y_det,'y','2dcnn_detangles')
'''
#plot_by_clustersize(residuals_x,residuals_y,'2dcnn',img_ext)
