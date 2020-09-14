#=================================
# Author: Sanjana Sekhar
# Date: 13 Sep 20
#=================================

#no of events to train on = 1230000
#no of events to test on = 1000000


import numpy as np
import h5py

#=====train files===== 

f = h5py.File("train_d49301_d49341.hdf5", "w")

n_files = 41

#30000 matrices per file
n_train = 30000*n_files

#"image" size = 13x21x1
train_data = np.zeros((n_train,13,21,1))
x_position = np.zeros((n_train,1))
y_position = np.zeros((n_train,1))
cosx = np.zeros((n_train,1))
cosy = np.zeros((n_train,1))
cosz = np.zeros((n_train,1))

n_events = 0

for i in range(1,n_files+1):

	train_out = open("template_events_d49301_d49341/template_events_d49%i.out"%(300+i), "r")
	#print("writing to file %i \n",i)
	lines = train_out.readlines()
	train_out.close()

	#delete first 2 lines		
	del lines[0:2]

	n=0
	position_data=[]

	for j in range(0,30000):

		#there are n 13x21 arrays in the file, extract each array 
		array2d = [[float(digit) for digit in line.split()] for line in lines[n+1:n+14]]
		one_train=np.array(array2d)
		#normalize the matrix 
		norm = np.linalg.norm(one_train)
		one_train = one_train/norm
		#reshape to (13,21,1) -> "image"
		train_data[j+n_events] = one_train[:,:,np.newaxis]

		#preceding each matrix is: x, y, z, cos x, cos y, cos z, nelec
		#cota = cos y/cos z ; cotb = cos x/cos z
		position_data = lines[n].split(' ')
		x_position[j+n_events] = float(position_data[0])
		y_position[j+n_events] = float(position_data[1])
		cosx[j+n_events] = float(position_data[3])
		cosy[j+n_events] = float(position_data[4])
		cosz[j+n_events] = float(position_data[5])

		n+=14

	n_events+=30000	


#IS IT BETTER TO SPECIFIY DTYPES?
train_dset = f.create_dataset("train_hits", np.shape(train_data), data=train_data)
x_train_dset = f.create_dataset("x", np.shape(x_position), data=x_position)
y_train_dset = f.create_dataset("y", np.shape(y_position), data=y_position)
cosx_train_dset = f.create_dataset("cosx", np.shape(cosx), data=cosx)
cosy_train_dset = f.create_dataset("cosy", np.shape(cosy), data=cosy)
cosz_train_dset = f.create_dataset("cosz", np.shape(cosz), data=cosz)

print("made train h5 file. no of events to train on = %i"%(n_train))
print("making test h5 file\n")

#======test files========

f = h5py.File("test_d49350.hdf5", "w")

n_test = 1000000

#"image" size = 13x21x1
test_data = np.zeros((n_test,13,21,1))
x_position = np.zeros((n_test,1))
y_position = np.zeros((n_test,1))
cosx = np.zeros((n_test,1))
cosy = np.zeros((n_test,1))
cosz = np.zeros((n_test,1))

test_out = open("template_events_d49350.out", "r")
#print("writing to file %i \n",i)
lines = test_out.readlines()
test_out.close()

#delete first 2 lines		
del lines[0:2]

n=0
position_data=[]

for j in range(0,n_test):

	#there are n 13x21 arrays in the file, extract each array 
	array2d = [[float(digit) for digit in line.split()] for line in lines[n+1:n+14]]
	test_data[j] = np.array(array2d)[:,:,np.newaxis] #reshape (13,21)->(13,21,1)

	#preceding each matrix is: x, y, z, cos x, cos y, cos z, nelec
	#cota = cos y/cos z ; cotb = cos x/cos z
	position_data = lines[n].split(' ')
	x_position[j] = float(position_data[0])
	y_position[j] = float(position_data[1])
	cosx[j] = float(position_data[3])
	cosy[j] = float(position_data[4])
	cosz[j] = float(position_data[5])

	n+=14
		
test_dset = f.create_dataset("test_hits", np.shape(test_data), data=test_data)
x_test_dset = f.create_dataset("x", np.shape(x_position), data=x_position)
y_test_dset = f.create_dataset("y", np.shape(y_position), data=y_position)
cosx_test_dset = f.create_dataset("cosx", np.shape(cosx), data=cosx)
cosy_test_dset = f.create_dataset("cosy", np.shape(cosy), data=cosy)
cosz_test_dset = f.create_dataset("cosz", np.shape(cosz), data=cosz)

print("made test h5 file. no of events to test on = %i"%(n_test))
