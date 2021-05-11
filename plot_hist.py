import numpy as np
import matplotlib.pyplot as plt

dx = np.genfromtxt("dx_1dcnn_new.txt")
x_axis = np.linspace(0,3000,100)

plt.hist(dx,bins=x_axis)
plt.title("dx = x_generic - x_1dcnn")
plt.xlabel("microns")
plt.savefig('dx_1dcnn.png')
