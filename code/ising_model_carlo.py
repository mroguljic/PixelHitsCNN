#===================================================================
# Metropolis algo for generating ising model
# Author: Sanjana Sekhar
# Date: 12/8/20
#===================================================================

import numpy as np
import numba 
from numba import jit
import numpy.random as rng
import matplotlib
import matplotlib.pyplot as plt

@jit
def monte_carlo_ising(Q,N,kT,lattice):

	ising = np.zeros((Q,N,N))
	mag = np.zeros((Q,1))
	accept = 0

	for index in range(0,2):
		E_i,E_f=0,0
		#generate a random no i and j for index of spin to be flipped
		i,j,r = rng.randint(0,N), rng.randint(0,N), rng.uniform(0,1)
		print('===============================')
		print(i,j,r)
		test_lattice = np.copy(lattice)
		#flip
		test_lattice[i,j] = -test_lattice[i,j]
		
		#Compute energy for both configs
		#numpy is modifying the original too
		#check right
		if(j!=N-1):
			print('right')
			E_i+=-(lattice[i,j]*lattice[i,j+1])
			E_f+=-(test_lattice[i,j]*test_lattice[i,j+1])
		#check left
		if(j!=0):
			print('left')
			E_i+=-(lattice[i,j]*lattice[i,j-1])
			E_f+=-(test_lattice[i,j]*test_lattice[i,j-1])
		#check top 
		if(i!=0):
			print('top')
			E_i+=-(lattice[i,j]*lattice[i-1,j])
			E_f+=-(test_lattice[i,j]*test_lattice[i-1,j])
		#check bottom
		if(i!=N-1):
			print('bottom')
			E_i+=-(lattice[i,j]*lattice[i+1,j])
			E_f+=-(test_lattice[i,j]*test_lattice[i+1,j])

		#make the choice 
		print('E_i = ',E_i)
		print(lattice)
		print('E_f = ',E_f)
		print(test_lattice)
		delE = E_f - E_i 
		print('delE = ',delE)
		if(delE < 0 or (delE >= 0 and r < np.exp(-delE/kT))):
			lattice = np.copy(test_lattice)
			ising[accept] = lattice
			#find magnetization
			#N_plus = np.sum(lattice.clip(0,1))
			#N_minus = N*N - N_plus
			#M = (N_plus - N_minus)/(N*N)
			mag[accept] = (2*np.sum(lattice.clip(0,1))-(N*N))/(N*N)
			accept+=1
	print('accept = ',accept)
	return ising[:accept],mag[:accept]

N = 10
Q = 10000
J = 1
kT = 1.5

#Start off with a random config
lattice = rng.choice([1, -1], size=(N, N))
print(lattice)
ising_config, mag = monte_carlo_ising(Q,N,kT,lattice)

plt.hist(mag, bins=np.arange(-2,2,0.1), histtype='step', density=True,linewidth=2)
plt.title('Probability of magnetization for T = %0.1f'%(kT))
plt.xlabel('magnetization')
#plt.show()


