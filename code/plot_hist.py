import numpy as np
import matplotlib.pyplot as plt

def plots_xy(results,label,algo):

	gen = results[:,0]*1e4
	nn = results[:,1]*1e4
	print(gen.shape,nn.shape)

	if label=='x':
		x = np.linspace(-0.8,0.8,15)
	else:
		x = np.linspace(-3,3,25)

	plt.hist(gen,bins=x)
	plt.title("%s_generic"%label)
	plt.xlabel("cm")
	plt.savefig('plots/%s_gen_%s.png'%(label,algo))
	plt.close()

	plt.hist(nn,bins=x)
	plt.title("%s_%s"%(label,algo))
	plt.xlabel("cm")
	plt.savefig('plots/%s_cnn_%s.png'%(label,algo))
	plt.close()

	plt.hist(gen-nn,bins=x)
	plt.title("d%s = %s_generic - %s_%s"%(label,label,label,algo))
	plt.xlabel("cm")
	plt.savefig('plots/%s_residuals_%s.png'%(label,algo))
	plt.close()


cnn1d_x = np.genfromtxt("txt_files/cnn_MC_x.txt")
cnn1d_y = np.genfromtxt("txt_files/cnn_MC_y.txt")
dnn_x = np.genfromtxt("txt_files/dnn_MC_x.txt")
dnn_y = np.genfromtxt("txt_files/dnn_MC_y.txt")

plots_xy(cnn1d_x,'x','cnn1d')
plots_xy(cnn1d_y,'y','cnn1d')
plots_xy(dnn_x,'x','cnn1d')
plots_xy(dnn_y,'y','cnn1d')

cnn2d = np.genfromtxt("txt_files/cnn2d_MC.txt")

x_gen = cnn2d[:,0]
y_gen = cnn2d[:,1]
x_cnn2d = cnn2d[:,2]
y_cnn2d = cnn2d[:,3]

cnn2d_x = np.vstack((x_gen,x_cnn2d)).T
cnn2d_y = np.vstack((y_gen,y_cnn2d)).T
print(cnn2d_x.shape, cnn2d_y.shape)

plots_xy(cnn2d_x,'x','cnn2d')
plots_xy(cnn2d_y,'y','cnn2d')
