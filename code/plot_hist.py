import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

img_ext = 'jun9'
SIMHITPERCLMAX = 10

def plot_residual(results, sim, label,algo):
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
	bins = np.linspace(-500,500,100)
	residuals = np.zeros_like(results)+9999

	for i in range(len(results)):
		for j in range(SIMHITPERCLMAX):
			if(abs(results[i]-sim[i][j])<residuals[i]):
				residuals[i] = (results[i]-sim[i][j])*1e4
	residuals = residuals[residuals<1000]
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
	plt.savefig("plots/%s_residuals_%s_%s.png"%(label,algo,img_ext))
	plt.close()


cnn1d_x = np.genfromtxt("txt_files/cnn_MC_x.txt")[:,1]
cnn1d_y = np.genfromtxt("txt_files/cnn_MC_y.txt")[:,1]

dnn_x = np.genfromtxt("txt_files/dnn_MC_x.txt")[:,1]
dnn_y = np.genfromtxt("txt_files/dnn_MC_y.txt")[:,1]

gen_x = np.genfromtxt("txt_files/cnn_MC_x.txt")[:,0]
gen_y = np.genfromtxt("txt_files/cnn_MC_y.txt")[:,0]

cnn2d = np.genfromtxt("txt_files/cnn2d_MC.txt")
cnn2d_x = cnn2d[:,2]
cnn2d_y = cnn2d[:,3]

simhits = np.genfromtxt("txt_files/simhits_MC.txt")
simhits_x = simhits[:,0:10]
simhits_y = simhits[:,10:20]
print(simhits_x.shape)
print(simhits_y.shape)

plot_residual(cnn1d_x,simhits_x,'x','1dcnn')
plot_residual(cnn1d_y,simhits_y,'y','1dcnn')
plot_residual(dnn_x,simhits_x,'x','dnn')
plot_residual(dnn_y,simhits_y,'y','dnn')
plot_residual(cnn2d_x,simhits_x,'x','2dcnn')
plot_residual(cnn2d_y,simhits_y,'y','2dcnn')
plot_residual(gen_x,simhits_x,'x','gen')
plot_residual(gen_y,simhits_y,'y','gen')
