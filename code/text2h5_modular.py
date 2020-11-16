#=================================
# Author: Sanjana Sekhar
# Date: 1 Nov 20
#=================================

import numpy as np
import h5py
import numpy.random as rng


def extract_matrices(lines,cluster_matrices):
	#delete first 2 lines
	pixelsize = lines[1] 		
	del lines[0:2]

	n=0

	n_per_file = int(len(lines)/14)

	for j in range(0,n_per_file):

		#there are n 13x21 arrays in the file, extract each array 
		array2d = [[float(digit) for digit in line.split()] for line in lines[n+1:n+14]]
		#reshape to (13,21,1) -> "image"
		#convert from pixelav sensor coords to normal coords
		cluster_matrices[j] = np.array(array2d).transpose()[:,:,np.newaxis]

		x_flat = cluster_matrices[j].reshape((21,13)).sum(axis=0)
		y_flat = cluster_matrices[j].reshape((21,13)).sum(axis=1)

		#possible issue: clustersize calc before noise added
		clustersize_x[j] = len(np.nonzero(x_flat)[0])
		clustersize_y[j] = len(np.nonzero(y_flat)[0])

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

	print("read in matrices from txt file\ntransposed all matrices")

def convert_pav_to_cms():
	
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

	print("converted labels from pixelav coords to cms coords \ncomputed cota cotb")

	return cota,cotb,x_position,y_position


def center_clusters(cluster_matrices):
	#shifting wav of cluster to matrix centre
	for index in np.arange(len(cluster_matrices)):

	#for index in np.arange(50):
		nonzero_list = np.transpose(np.asarray(np.nonzero(cluster_matrices[index])))
		nonzero_elements = cluster_matrices[index][np.nonzero(cluster_matrices[index])]
		#print(nonzero_elements.shape)
		nonzero_i = nonzero_list[:,0]-10. #x indices
		#print(nonzero_i.shape)
		nonzero_j = nonzero_list[:,1]-6. #y indices
		shift_i = -int(round(np.dot(nonzero_i,nonzero_elements)/np.sum(nonzero_elements)))
		shift_j = -int(round(np.dot(nonzero_j,nonzero_elements)/np.sum(nonzero_elements)))
		
		if(shift_i>0 and np.amax(nonzero_i)!=20):
			#shift down iff there is no element at the last column
			cluster_matrices[index] = np.roll(cluster_matrices[index],shift_i,axis=0)
			#shift hit position too
			y_position[index]-=pixelsize_y[index]*shift_i
		if(shift_i<0 and np.amin(nonzero_i)!=0):
			#shift up iff there is no element at the first column
			cluster_matrices[index] = np.roll(cluster_matrices[index],shift_i,axis=0)
			#shift hit position too
			y_position[index]-=pixelsize_y[index]*shift_i
		if(shift_j>0 and np.amax(nonzero_j)!=12):
			#shift right iff there is no element in the last row
			cluster_matrices[index] = np.roll(cluster_matrices[index],shift_j,axis=1)
			#shift hit position too
			x_position[index]+=pixelsize_x[index]*shift_j
		if(shift_j<0 and np.amin(nonzero_j)!=0):
			#shift left iff there is no element in the first row
			cluster_matrices[index] = np.roll(cluster_matrices[index],shift_j,axis=1)
			#shift hit position too
			x_position[index]+=pixelsize_x[index]*shift_j

	print("shifted wav of clusters to matrix centres")

def apply_noise(cluster_matrices,fe_type):
	#add 2 types of noise

	if(fe_type==1): #linear gain
		for index in np.arange(len(cluster_matrices)):
			noise_1 = rng.normal(loc=0.,scale=1.,size=(21*13)).reshape((21,13,1)) #generate a matrix with 21x13 elements from a gaussian dist with mu = 0 and sig = 1
			noise_2 = rng.normal(loc=0.,scale=1.,size=(21*13)).reshape((21,13,1))
			cluster_matrices[index]+= gain_frac*noise_1*cluster_matrices[index] + readout_noise*noise_2
		print("applied linear gain")

	elif(fe_type==2): #tanh gain
		for index in np.arange(len(cluster_matrices)):
			noise_1 = rng.normal(loc=0.,scale=1.,size=(21*13)).reshape((21,13,1)) #generate a matrix with 21x13 elements from a gaussian dist with mu = 0 and sig = 1
			noise_2 = rng.normal(loc=0.,scale=1.,size=(21*13)).reshape((21,13,1))
			adc = (float)((int)(p3+p2*tanh(p0*(cluster_matrices[index] + vcaloffst)/(7.0*vcal) - p1)))
			cluster_matrices[index] = ((float)((1.+gain_frac*noise_1)*(vcal*gain*(adc-ped))) - vcaloffst + noise_2*readout_noise)
		print("applied tanh gain")

def apply_threshold(cluster_matrices,threshold):
	#if n_elec < 1000 -> 0
	below_threshold_i = cluster_matrices < threshold
	cluster_matrices[below_threshold_i] = 0
	print("applied threshold")

def project_matrices_xy(cluster_matrices):

	#for dnn
	for index in np.arange(len(cluster_matrices)):
		x_flat[index] = cluster_matrices[index].reshape((21,13)).sum(axis=0)
		y_flat[index] = cluster_matrices[index].reshape((21,13)).sum(axis=1)

	print('took x and y projections of all matrices')	


def create_datasets(f,cluster_matrices,x_flat,y_flat,dset_type):
	#IS IT BETTER TO SPECIFIY DTYPES?
	clusters_dset = f.create_dataset("%s_hits"%(dset_type), np.shape(cluster_matrices), data=cluster_matrices)
	x_dset = f.create_dataset("x", np.shape(x_position), data=x_position)
	y_dset = f.create_dataset("y", np.shape(y_position), data=y_position)
	cota_dset = f.create_dataset("cota", np.shape(cota), data=cota)
	cotb_dset = f.create_dataset("cotb", np.shape(cotb), data=cotb)
	clustersize_x_dset = f.create_dataset("clustersize_x", np.shape(clustersize_x), data=clustersize_x)
	clustersize_y_dset = f.create_dataset("clustersize_y", np.shape(clustersize_y), data=clustersize_y)
	x_flat_dset = f.create_dataset("%s_x_flat"%(dset_type), np.shape(x_flat), data=x_flat)
	y_flat_dset = f.create_dataset("%s_y_flat"%(dset_type), np.shape(y_flat), data=y_flat)

	print("made %s h5 file. no. of events to %s on: %i"%(dset_type,dset_type,len(cluster_matrices)))

fe_type = 1
gain_frac     = 0.08;
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

date = "nov16"
filename = "full_angle_scan"

#=====train files===== 

print("making train h5 file")


train_out = open("templates/template_events_d99352.out", "r")
#print("writing to file %i \n",i)
lines = train_out.readlines()
train_out.close()

n_train = int((len(lines)-2)/14)
print("n_train = ",n_train)

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
clustersize_x = np.zeros((n_train,1))
clustersize_y = np.zeros((n_train,1))
x_flat = np.zeros((n_train,13))
y_flat = np.zeros((n_train,21))

extract_matrices(lines,train_data)
#print(train_data[0].reshape((21,13)))
cota,cotb,x_position,y_position = convert_pav_to_cms()
#print(x_position_pav[0],y_position_pav[0])
#print(x_position[0],y_position[0])
center_clusters(train_data)
#print(train_data[0].reshape((21,13)))
#print(x_position[0],y_position[0])
#n_elec were scaled down by 10 so multiply
train_data = 10*train_data
print("multiplied all elements by 10")
#print(train_data[0].reshape((21,13)))

apply_noise(train_data,fe_type)
#print(train_data[0].reshape((21,13)))
apply_threshold(train_data,threshold)
#print(train_data[0].reshape((21,13)))
project_matrices_xy(train_data)
#print(x_flat[0],y_flat[0])
#print(clustersize_x[0],clustersize_y[0])

f = h5py.File("h5_files/train_%s_%s.hdf5"%(filename,date), "w")

create_datasets(f,train_data,x_flat,y_flat,"train")

#====== test files ========

print("making test h5 file.")

test_out = open("templates/template_events_d99353.out", "r")
#print("writing to file %i \n",i)
lines = test_out.readlines()
test_out.close()

n_test = int((len(lines)-2)/14)
print("n_test = ",n_test)

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
clustersize_x = np.zeros((n_test,1))
clustersize_y = np.zeros((n_test,1))
x_flat = np.zeros((n_test,13))
y_flat = np.zeros((n_test,21))

extract_matrices(lines,test_data)
#print(test_data[0].reshape((21,13)))
cota,cotb,x_position,y_position = convert_pav_to_cms()
#print(x_position_pav[0],y_position_pav[0])
#print(x_position[0],y_position[0])
center_clusters(test_data)
#print(test_data[0].reshape((21,13)))
#print(x_position[0],y_position[0])
#n_elec were scaled down by 10 so multiply
test_data = 10*test_data
print("multiplied all elements by 10")
#print(test_data[0].reshape((21,13)))

apply_noise(test_data,fe_type)
#print(test_data[0].reshape((21,13)))
apply_threshold(test_data,threshold)
#print(test_data[0].reshape((21,13)))
project_matrices_xy(test_data)
#print(x_flat[0],y_flat[0])
#print(clustersize_x[0],clustersize_y[0])

f = h5py.File("h5_files/test_%s_%s.hdf5"%(filename,date), "w")

create_datasets(f,test_data,x_flat,y_flat,"test")



