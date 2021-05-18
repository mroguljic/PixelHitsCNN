import numpy as np
import matplotlib.pyplot as plt

dx = np.genfromtxt("txt_files/cnn_test.txt")

x_axis = np.linspace(0,3000,100)

plt.hist(dx)
plt.title("generic output")
plt.xlabel("microns")
plt.savefig('plots/cnn_test.png')

print("Max position: %f microns; Min position: %f microns"%(np.amax(dx)*1e4,np.amin(dx)*1e4))