import numpy as np
import matplotlib.pyplot as plt

def plots_xy(results,label,algo):

	gen = results[:,0]*1e4
	nn = results[:,1]*1e4

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

	bins = np.linspace(-1700,1700,100)
	plt.hist(gen-nn,bins=bins)
	plt.title("d%s = %s_generic - %s_%s"%(label,label,label,algo))
	plt.xlabel("microns")
	plt.savefig('plots/%s_residuals_%s.png'%(label,algo))
	plt.close()


cnn1d_x = np.genfromtxt("txt_files/cnn_MC_x.txt")
cnn1d_y = np.genfromtxt("txt_files/cnn_MC_y.txt")
dnn_x = np.genfromtxt("txt_files/dnn_MC_x.txt")
dnn_y = np.genfromtxt("txt_files/dnn_MC_y.txt")

plots_xy(cnn1d_x,'x','1dcnn')
plots_xy(cnn1d_y,'y','1dcnn')
plots_xy(dnn_x,'x','dnn')
plots_xy(dnn_y,'y','dnn')

cnn2d = np.genfromtxt("txt_files/cnn2d_MC.txt")

x_gen = cnn2d[:,0]
y_gen = cnn2d[:,1]
x_cnn2d = cnn2d[:,2]
y_cnn2d = cnn2d[:,3]

cnn2d_x = np.vstack((x_gen,x_cnn2d)).T
cnn2d_y = np.vstack((y_gen,y_cnn2d)).T
print(cnn2d_x.shape, cnn2d_y.shape)

plots_xy(cnn2d_x,'x','2dcnn')
plots_xy(cnn2d_y,'y','2dcnn')
