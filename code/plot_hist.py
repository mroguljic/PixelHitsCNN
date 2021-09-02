import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages

img_ext = '090221'
SIMHITPERCLMAX = 10

def plot_residual(residuals,label,algo):
	'''
	results = results*1e4
	sim = results[:,1]*1e4

	plt.hist(nn,facecolor='None',edgecolor='r',lw=2,label="%s_%s"%(label,algo),bins=30)
	#plt.title()
	#plt.xlabel("microns")

	plt.hist(gen,facecolor='None',edgecolor='b',lw=2,label="%s_generic"%label,bins=30)
	plt.title("Comparison between %s_generic and %s_%s outputs"%(label,label,algo))
	plt.xlabel("microns")
	#plt.savefig('plots/%s_gen_%s.png'%(label,algo))
	#plt.close()

	
	plt.legend()
	plt.savefig('plots/%s_compare_%s.png'%(label,algo))
	plt.close()
	'''
	bins = np.linspace(-300,300,100)
	residuals*=1e4
	print("====== %s %s ======"%(algo,label))
	'''
	residuals = np.zeros_like(results)+99999.

	for i in range(len(results)):
		for j in range(SIMHITPERCLMAX):
			if(abs(results[i]-sim[i][j])<residuals[i]):
				#if i==16049: 
				#	print("idx 16049: ",results[i], sim[i][j],abs(results[i]-sim[i][j]),residuals[i])
				residuals[i] = (results[i]-sim[i][j])

			#else: print("abs(results[i]-sim[i][j])<residuals[i]",results[i])
	
	
	#print("no of residuals >1000: ",len(np.argwhere(residuals>1000)))
	#print(np.argwhere(residuals>1000),residuals[residuals>1000])
	#residuals = residuals[residuals<1000]
	'''
	RMS = np.sqrt(np.mean(residuals*residuals))
	mean, sigma = norm.fit(residuals)

	plt.hist(residuals, bins=bins, histtype='step', density=True,linewidth=2,label=r'$\vartriangle$'+label)
	xmin, xmax = plt.xlim()
	x = np.linspace(xmin, xmax, 100)
	p = norm.pdf(x, mean, sigma)
	plt.title('%s - residuals in %s, RMS = %0.2f, %s = %0.2f'%(algo, label,RMS,r'$\sigma$',sigma))
	#plt.ylabel('No. of samples')
	plt.xlabel(r'$\mu m$')

	plt.plot(x, p, 'k', linewidth=1,color='red',label='gaussian fit')
	plt.legend()
	plt.savefig("plots/CMSSW/residuals/%s_residuals_%s_%s.png"%(label,algo,img_ext))
	plt.close()

	

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

cnn1d_x = np.genfromtxt("txt_files/cnn1d_MC_x.txt")
cnn1d_y = np.genfromtxt("txt_files/cnn1d_MC_y.txt")

cnn1d_x_det = np.genfromtxt("txt_files/cnn1d_MC_x_detangles.txt")
cnn1d_y_det = np.genfromtxt("txt_files/cnn1d_MC_y_detangles.txt")

cnn2d_x = np.genfromtxt("txt_files/cnn2d_MC_x.txt")
cnn2d_y = np.genfromtxt("txt_files/cnn2d_MC_y.txt")

cnn2d_x_det = np.genfromtxt("txt_files/cnn2d_MC_x_detangles.txt")
cnn2d_y_det = np.genfromtxt("txt_files/cnn2d_MC_y_detangles.txt")

gen_x_det = np.genfromtxt("txt_files/generic_MC_x_detangles.txt")
gen_y_det = np.genfromtxt("txt_files/generic_MC_y_detangles.txt")

gen_x = np.genfromtxt("txt_files/generic_MC_x.txt")
gen_y = np.genfromtxt("txt_files/generic_MC_y.txt")

template_x = np.genfromtxt("txt_files/template_MC_x.txt")
template_y = np.genfromtxt("txt_files/template_MC_y.txt")

simhits_x = np.genfromtxt("txt_files/simhits_MC_x.txt")
simhits_y = np.genfromtxt("txt_files/simhits_MC_y.txt")

simhits_x_gen = np.genfromtxt("txt_files/simhits_MC_x_gendet.txt")
simhits_y_gen = np.genfromtxt("txt_files/simhits_MC_y_gendet.txt")

print("gen_x shape = ",gen_x.shape)
print("template_x shape = ",template_x.shape)
print("cnn1d_x shape = ",cnn1d_x.shape)
print("cnn2d_x shape = ",cnn2d_x.shape)

residuals_x = plot_residual(cnn1d_x,'x','1dcnn')
#residuals_y = plot_residual(cnn1d_y,simhits_y,'y','1dcnn')

#residuals_x = plot_residual(cnn1d_x_det,simhits_x,'x','1dcnn_detangles')
#residuals_y = plot_residual(cnn1d_y_det,simhits_y,'y','1dcnn_detangles')
#plot_by_clustersize(residuals_x,residuals_y,'1dcnn',img_ext)

#residuals_x = plot_residual(dnn_x,simhits_x,'x','dnn')
#residuals_y = plot_residual(dnn_y,simhits_y,'y','dnn')

residuals_x = plot_residual(gen_x,'x','gen')
#residuals_y = plot_residual(gen_y,simhits_y,'y','gen')

#residuals_x = plot_residual(gen_x_det,simhits_x_gen,'x','gen_detangles')
#residuals_y = plot_residual(gen_y_det,simhits_y_gen,'y','gen_detangles')

residuals_x = plot_residual(template_x,'x','template')
#residuals_y = plot_residual(template_y,simhits_y,'y','template')

residuals_x = plot_residual(cnn2d_x,'x','2dcnn')
#residuals_y = plot_residual(cnn2d_y,simhits_y,'y','2dcnn')

#residuals_x = plot_residual(cnn2d_x_det,simhits_x,'x','2dcnn_detangles')
#residuals_y = plot_residual(cnn2d_y_det,simhits_y,'y','2dcnn_detangles')

#plot_by_clustersize(residuals_x,residuals_y,'2dcnn',img_ext)


