import numpy as np
import matplotlib.pyplot as plt

dx = np.genfromtxt("dy_1dcnn_new.txt")
x_axis = np.linspace(0,1000,100)

plt.hist(dx,bins=x_axis)
plt.title("dy = y_generic - y_1dcnn")
plt.xlabel("microns")
plt.savefig('dy_1dcnn.png')
