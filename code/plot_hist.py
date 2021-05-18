import numpy as np
import matplotlib.pyplot as plt

dx = np.genfromtxt("txt_files/dx_1dcnn.txt")

x_axis = np.linspace(0,3000,100)

plt.hist(dx)
plt.title("dx = x_generic - x_1dcnn")
plt.xlabel("cms")
plt.savefig('plots/dx_cms.png')
plt.close()
#print("Max position: %f microns; Min position: %f microns"%(np.amax(dx)*1e4,np.amin(dx)*1e4))

dx = np.genfromtxt("txt_files/generic_test.txt")

x_axis = np.linspace(0,3000,100)

plt.hist(dx)
plt.title("output from the generic algo")
plt.xlabel("cms")
plt.savefig('plots/generic_test.png')
plt.close()

dx = np.genfromtxt("txt_files/cnn_test.txt")

x_axis = np.linspace(0,3000,100)

plt.hist(dx)
plt.title("output from the cnn algo:")
plt.xlabel("cms")
plt.savefig('plots/cnn_test.png')
plt.close()