#=================================
# Author: Sanjana Sekhar
# Date: 13 Sep 20
#=================================

import numpy as np
import h5py

fe_type = 1
gain_frac     = 0.;
readout_noise = 350.;

#--- Variables we can change, but we start with good default values
vcal = 47.0;	
vcaloffst = 60.0;

#--- PhaseII - initial guess
threshold = 1000; # threshold in e-
qperToT = 1500; # e- per TOT
nbitsTOT = 4; # fixed and carved in stone?
ADCMax = np.power(2, nbitsTOT)-1;
dualslope = 4;

#--- Constants (could be made variables later)
gain  = 3.19;
ped   = 16.46;
p0    = 0.01218;
p1    = 0.711;
p2    = 203.;
p3    = 148.;	

date = "oct11_by10"

#=====train files===== 

f = h5py.File("h5_files/train_d49301_d49341_%s.hdf5"%(date), "w")


n_per_file = 30000
n_files = 41


#no of events to train on = 1230000
#no of events to test on = 1000000
#30000 matrices per file
n_train = n_per_file*n_files

#"image" size = 13x21x1
train_data = np.zeros((n_train,21,13,1))
x_position_pav = np.zeros((n_train,1))
y_position_pav = np.zeros((n_train,1))
cosx = np.zeros((n_train,1))
cosy = np.zeros((n_train,1))
cosz = np.zeros((n_train,1))
pixelsize_x = np.zeros((n_train,1))
pixelsize_y = np.zeros((n_train,1))
pixelsize_z = np.zeros((n_train,1))

n_events = 0

for i in range(1,n_files+1):

	train_out = open("templates/template_events_d49301_d49341/template_events_d49%i.out"%(300+i), "r")
	#print("writing to file %i \n",i)
	lines = train_out.readlines()
	train_out.close()

	#delete first 2 lines
	pixelsize = lines[1] 		
	del lines[0:2]

	n=0

	for j in range(0,n_per_file):

		#there are n 13x21 arrays in the file, extract each array 
		array2d = [[float(digit) for digit in line.split()] for line in lines[n+1:n+14]]
		#reshape to (13,21,1) -> "image"
		#convert from pixelav sensor coords to normal coords
		train_data[j+n_events] = np.array(array2d).transpose()[:,:,np.newaxis]

		#preceding each matrix is: x, y, z, cos x, cos y, cos z, nelec
		#cota = cos y/cos z ; cotb = cos x/cos z
		position_data = lines[n].split(' ')
		x_position_pav[j+n_events] = float(position_data[0])
		y_position_pav[j+n_events] = float(position_data[1])
		cosx[j+n_events] = float(position_data[3])
		cosy[j+n_events] = float(position_data[4])
		cosz[j+n_events] = float(position_data[5])

		pixelsize_data = pixelsize.split('  ')
		pixelsize_x[j+n_events] = float(pixelsize_data[1]) #flipped on purpose cus matrix has transposed
		pixelsize_y[j+n_events] = float(pixelsize_data[0])
		pixelsize_z[j+n_events] = float(pixelsize_data[2])

		n+=14

	n_events+=n_per_file	
#============= preprocessing =====================
#switching out of pixelav coords to localx and localy
#remember that h5 files have already been made with transposed matrices
'''
float z_center = zsize/2.0;
float xhit = x1 + (z_center - z1) * cosx/cosz; cosx/cosz = cotb
float yhit = y1 + (z_center - z1) * cosy/cosz; cosy/cosz = cota
x -> -y
y -> -x
z1 is always 0 
'''
cota = cosy/cosz
cotb = cosx/cosz
x_position = -(y_position_pav + (pixelsize_z/2.)*cota)
y_position = -(x_position_pav + (pixelsize_z/2.)*cotb)

print("transposed all train matrices\nconverted train_labels from pixelav coords to cms coords \ncomputed train cota cotb\n")


#shifting central hit away from matrix centre
for index in np.arange(len(train_data)):
	nonzero_list = np.asarray(np.nonzero(train_data[index]))
	nonzero_i = np.sort(nonzero_list[0,:])
	nonzero_j = np.sort(nonzero_list[1,:])
	if(index%4 == 0 and nonzero_j[0]!=0):
		#shift left iff there is no element at 0th column
		train_data[index] = np.roll(train_data[index],-1,axis=1)
		#shift hit position too
		y_position[index]-=pixelsize_y[index]

	if(index%4 == 1 and nonzero_j[-1]!=12):
		#shift right iff there is no element at the last column
		train_data[index] = np.roll(train_data[index],1,axis=1)
		#shift hit position too
		y_position[index]+=pixelsize_y[index]

	if(index%4 == 2 and nonzero_i[0]!=0):
		#shift up iff there is no element in the 0th row
		train_data[index] = np.roll(train_data[index],-1,axis=0)
		#shift hit position too
		x_position[index]+=pixelsize_x[index]

	if(index%4 == 3 and nonzero_i[-1]!=20):
		#shift down iff there is no element in the last row
		train_data[index] = np.roll(train_data[index],1,axis=0)
		#shift hit position too
		x_position[index]-=pixelsize_x[index]

print("shifted pixel hits away from matrix centre")

#n_elec were scaled down by 10 so multiply
train_data = 10*train_data 

print("multiplied all elements by 10")

#add 2 types of noise

if(fe_type==1): #linear gain
	for index in np.arange(len(train_data)):
		noise = np.random.normal(0,1,(21*13)).reshape((21,13,1)) #generate a matrix with 21x13 elements from a gaussian dist
		train_data[index]+= gain_frac*noise*train_data[index] + readout_noise*noise
	print("applied linear gain")

elif(fe_type==2): #tanh gain
	for index in np.arange(len(train_data)):
		noise = np.random.normal(0,1,(21*13)).reshape((21,13,1))
		adc = (float)((int)(p3+p2*tanh(p0*(train_data[index] + vcaloffst)/(7.0*vcal) - p1)))
		train_data[index] = ((float)((1.+gain_frac*noise)*(vcal*gain*(adc-ped))) - vcaloffst + noise*readout_noise)
	print("applied tanh gain")


#if n_elec < 1000 -> 0
below_threshold_i = train_data < threshold
train_data[below_threshold_i] = 0
print("applied threshold")

#train_data = (train_data/10)#.astype('int32')
print("divided by 10")


#IS IT BETTER TO SPECIFIY DTYPES?
train_dset = f.create_dataset("train_hits", np.shape(train_data), data=train_data)
x_train_dset = f.create_dataset("x", np.shape(x_position), data=x_position)
y_train_dset = f.create_dataset("y", np.shape(y_position), data=y_position)
cota_train_dset = f.create_dataset("cota", np.shape(cota), data=cota)
cotb_train_dset = f.create_dataset("cotb", np.shape(cotb), data=cotb)

print("made train h5 file. no of events to train on = %i\n"%(n_train))
print("making test h5 file\n")

#-----------------------------------------------------------------------
#====== test files ========
#-----------------------------------------------------------------------

f = h5py.File("h5_files/test_d49350_%s.hdf5"%(date), "w")

n_test = 1000000

#"image" size = 13x21x1
test_data = np.zeros((n_test,21,13,1))
x_position_pav = np.zeros((n_test,1))
y_position_pav = np.zeros((n_test,1))
cosx = np.zeros((n_test,1))
cosy = np.zeros((n_test,1))
cosz = np.zeros((n_test,1))
pixelsize_x = np.zeros((n_test,1))
pixelsize_y = np.zeros((n_test,1))
pixelsize_z = np.zeros((n_test,1))


test_out = open("templates/template_events_d49350.out", "r")
#print("writing to file %i \n",i)
lines = test_out.readlines()
test_out.close()

#delete first 2 lines
pixelsize = lines[1]		
del lines[0:2]

n=0

for j in range(0,n_test):

	#there are n 13x21 arrays in the file, extract each array 
	array2d = [[float(digit) for digit in line.split()] for line in lines[n+1:n+14]]
	#reshape (13,21)->(13,21,1)
	#convert from pixelav sensor coords to normal coords
	test_data[j] = np.array(array2d).transpose()[:,:,np.newaxis]

	#preceding each matrix is: x, y, z, cos x, cos y, cos z, nelec
	#cota = cos y/cos z ; cotb = cos x/cos z
	position_data = lines[n].split(' ')
	x_position_pav[j] = float(position_data[0])
	y_position_pav[j] = float(position_data[1])
	cosx[j] = float(position_data[3])
	cosy[j] = float(position_data[4])
	cosz[j] = float(position_data[5])

	pixelsize_data = pixelsize.split('  ')
	pixelsize_x[j] = float(pixelsize_data[1]) #flipped on purpose cus matrix has transposed
	pixelsize_y[j] = float(pixelsize_data[0])
	pixelsize_z[j] = float(pixelsize_data[2])

	n+=14


#============= preprocessing =====================
#switching out of pixelav coords to localx and localy
#remember that h5 files have already been made with transposed matrices

#float z_center = zsize/2.0;
#float xhit = x1 + (z_center - z1) * cosx/cosz; cosx/cosz = cotb
#float yhit = y1 + (z_center - z1) * cosy/cosz; cosy/cosz = cota
#x -> -y
#y -> -x
#z1 is always 0 

cota = cosy/cosz
cotb = cosx/cosz
x_position = -(y_position_pav + (pixelsize_z/2.)*cota)
y_position = -(x_position_pav + (pixelsize_z/2.)*cotb)

print("transposed all test matrices\nconverted test_labels from pixelav coords to cms coords \ncomputed test cota cotb\n")

#shifting central hit away from matrix centre
for index in np.arange(len(test_data)):
	nonzero_list = np.asarray(np.nonzero(test_data[index]))
	nonzero_i = np.sort(nonzero_list[0,:])
	nonzero_j = np.sort(nonzero_list[1,:])
	if(index%4 == 0 and nonzero_j[0]!=0):
		#shift left iff there is no element at 0th column
		test_data[index] = np.roll(test_data[index],-1,axis=1)
		#shift hit position too
		y_position[index]-=pixelsize_y[index]

	if(index%4 == 1 and nonzero_j[-1]!=12):
		#shift right iff there is no element at the last column
		test_data[index] = np.roll(test_data[index],1,axis=1)
		#shift hit position too
		y_position[index]+=pixelsize_y[index]

	if(index%4 == 2 and nonzero_i[0]!=0):
		#shift up iff there is no element in the 0th row
		test_data[index] = np.roll(test_data[index],-1,axis=0)
		#shift hit position too
		x_position[index]+=pixelsize_x[index]

	if(index%4 == 3 and nonzero_i[-1]!=20):
		#shift down iff there is no element in the last row
		test_data[index] = np.roll(test_data[index],1,axis=0)
		#shift hit position too
		x_position[index]-=pixelsize_x[index]

print("shifted pixel hits away from matrix centre")

#n_elec were scaled down by 10 so multiply
test_data = 10*test_data 

print("multiplied all elements by 10")

#add 2 types of noise

if(fe_type==1): #linear gain
	for index in np.arange(len(test_data)):
		noise = np.random.normal(0,1,(21*13)).reshape((21,13,1)) #generate a matrix with 21x13 elements from a gaussian dist
		test_data[index]+= gain_frac*noise*test_data[index] + readout_noise*noise
	print("applied linear gain")

elif(fe_type==2): #tanh gain
	for index in np.arange(len(test_data)):
		adc = (float)((int)(p3+p2*tanh(p0*(test_data[index] + vcaloffst)/(7.0*vcal) - p1)))
		test_data[index] = ((float)((1.+gain_frac*noise)*(vcal*gain*(adc-ped))) - vcaloffst + noise*readout_noise)
	print("applied tanh gain")

#if n_elec < 1000 -> 0
below_threshold_i = test_data < threshold
test_data[below_threshold_i] = 0
print("applied threshold")

#test_data = (test_data/10)#.astype('int32')
print("divided by 10")

#IS IT BETTER TO SPECIFIY DTYPES?
test_dset = f.create_dataset("test_hits", np.shape(test_data), data=test_data)
x_test_dset = f.create_dataset("x", np.shape(x_position), data=x_position)
y_test_dset = f.create_dataset("y", np.shape(y_position), data=y_position)
cota_test_dset = f.create_dataset("cota", np.shape(cota), data=cota)
cotb_test_dset = f.create_dataset("cotb", np.shape(cotb), data=cotb)

print("made test h5 file. no of events to test on = %i"%(n_test))
