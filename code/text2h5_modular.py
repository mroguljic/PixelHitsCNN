#=================================
# Author: Sanjana Sekhar
# Date: 1 Nov 20
#=================================

import numpy as np
import h5py
import numpy.random as rng
from skimage.measure import label


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
		one_mat = np.array(array2d)
		one_mat = np.flip(one_mat,0)
		one_mat = np.flip(one_mat,1)
		cluster_matrices[j]=one_mat[:,:,np.newaxis]

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

	print("read in matrices from txt file\nflipped all matrices")

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
	x_position = -(x_position_pav + (pixelsize_z/2.)*cota)
	y_position = -(y_position_pav + (pixelsize_z/2.)*cotb)

	print("converted labels from pixelav coords to cms coords \ncomputed cota cotb")

	return cota,cotb,x_position,y_position

def apply_noise(cluster_matrices,fe_type):
	#add 2 types of noise

	if(fe_type==1): #linear gain
		for index in np.arange(len(cluster_matrices)):
			noise_1 = rng.normal(loc=0.,scale=1.,size=(13*21)).reshape((13,21,1)) #generate a matrix with 21x13 elements from a gaussian dist with mu = 0 and sig = 1
			noise_2 = rng.normal(loc=0.,scale=1.,size=(13*21)).reshape((13,21,1))
			cluster_matrices[index]+= gain_frac*noise_1*cluster_matrices[index] + readout_noise*noise_2
		print("applied linear gain")

	elif(fe_type==2): #tanh gain
		for index in np.arange(len(cluster_matrices)):
			noise_1 = rng.normal(loc=0.,scale=1.,size=(13*21)).reshape((13,21,1)) #generate a matrix with 21x13 elements from a gaussian dist with mu = 0 and sig = 1
			noise_2 = rng.normal(loc=0.,scale=1.,size=(13*21)).reshape((13,21,1))
			adc = (float)((int)(p3+p2*tanh(p0*(cluster_matrices[index] + vcaloffst)/(7.0*vcal) - p1)))
			cluster_matrices[index] = ((float)((1.+gain_frac*noise_1)*(vcal*gain*(adc-ped))) - vcaloffst + noise_2*readout_noise)
		print("applied tanh gain")

def apply_threshold(cluster_matrices,threshold):
	#if n_elec < 1000 -> 0
	below_threshold_i = cluster_matrices < threshold
	cluster_matrices[below_threshold_i] = 0
	cluster_matrices=(cluster_matrices/10.).astype(int)
	print("applied threshold")
	return cluster_matrices


def center_clusters(cluster_matrices):
	
	for index in np.arange(len(cluster_matrices)):

	#for index in np.arange(50):
		#find clusters
		one_mat = cluster_matrices[index].reshape((13,21))
		#find connected components 
		labels = label(one_mat.clip(0,1))
		#find no of clusters
		n_clusters = np.amax(labels)
		max_cluster_size=0
		#if there is more than 1 cluster, the largest one is the main one
		if(n_clusters>1):
			for i in range(1,n_clusters+1):
				cluster_idxs_x = np.argwhere(labels==i)[:,0]
				cluster_idxs_y = np.argwhere(labels==i)[:,1]
				cluster_size = len(cluster_idxs_x)
				if cluster_size>max_cluster_size:
					max_cluster_size = cluster_size
					largest_idxs_x = cluster_idxs_x
					largest_idxs_y = cluster_idxs_y
				#if there are 2 clusters of the same size then the largest hit is the main one
				if cluster_size==max_cluster_size: #eg. 2 clusters of size 2
					if(np.amax(one_mat[largest_idxs_x,largest_idxs_y])<np.amax(one_mat[cluster_idxs_x,cluster_idxs_y])):
						largest_idxs_x = cluster_idxs_x
						largest_idxs_y = cluster_idxs_y
		else:
			largest_idxs_x = np.argwhere(labels==1)[:,0]
			largest_idxs_y = np.argwhere(labels==1)[:,1]
		
		#find clustersize
		clustersize_x[index] = int(len(np.unique(largest_idxs_x)))
		clustersize_y[index] = int(len(np.unique(largest_idxs_y)))

		#find geometric centre of the main cluster using avg
		center_x = int(np.mean(largest_idxs_x))
		center_y = int(np.mean(largest_idxs_y))
		#if the geometric centre is not at (7,11) shift cluster
		nonzero_list = np.asarray(np.nonzero(one_mat))
		nonzero_x = nonzero_list[0,:]
		nonzero_y = nonzero_list[1,:]
		if(center_x<6):
			#shift down
			shift = 6-center_x
			if(np.amax(nonzero_x)+shift<=12):
				one_mat=np.roll(one_mat,shift,axis=0)
				x_position[index]-=pixelsize_x[index]*shift

		if(center_x>6):
			#shift up
			shift = center_x-6
			if(np.amin(nonzero_x)-shift>=0):
				one_mat=np.roll(one_mat,-shift,axis=0)
				x_position[index]+=pixelsize_x[index]*shift

		if(center_y<10):
			#shift right
			shift = 10-center_y
			if(np.amax(nonzero_y)+shift<=20):
				one_mat=np.roll(one_mat,shift,axis=1)
				y_position[index]+=pixelsize_y[index]*shift

		if(center_y>10):
			#shift left
			shift = center_y-10
			if(np.amin(nonzero_y)-shift>=0):
				one_mat=np.roll(one_mat,-shift,axis=1)
				y_position[index]-=pixelsize_y[index]*shift

		cluster_matrices[index]=one_mat[:,:,np.newaxis]

	print("shifted centre of clusters to matrix centres")


def project_matrices_xy(cluster_matrices):

	#for dnn
	for index in np.arange(len(cluster_matrices)):
		x_flat[index] = cluster_matrices[index].reshape((13,21)).sum(axis=1)
		y_flat[index] = cluster_matrices[index].reshape((13,21)).sum(axis=0)

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

date = "nov30"
filename = "subset"

#=====train files===== 

#print("making train h5 file")


train_out = open("templates/template_events_d99353_subset.out", "r")
##print("writing to file %i \n",i)
lines = train_out.readlines()
train_out.close()

n_train = int((len(lines)-2)/14)
#print("n_train = ",n_train)

#"image" size = 13x21x1
train_data = np.zeros((n_train,13,21,1))
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
#print(train_data[0].reshape((13,21)))
cota,cotb,x_position,y_position = convert_pav_to_cms()
#print(x_position_pav[0],y_position_pav[0])
#print(x_position[0],y_position[0])

#n_elec were scaled down by 10 so multiply
train_data = 10*train_data
#print("multiplied all elements by 10")
#print(train_data[0].reshape((13,21)))

apply_noise(train_data,fe_type)
#print(train_data[0].reshape((13,21)))
train_data = apply_threshold(train_data,threshold)
#print(train_data[0].reshape((13,21)))

center_clusters(train_data)
#print(train_data[0].reshape((13,21)))
#print(x_position[0],y_position[0])

project_matrices_xy(train_data)
#print(x_flat[0],y_flat[0])
#print(clustersize_x[0],clustersize_y[0])

f = h5py.File("h5_files/train_%s_%s.hdf5"%(filename,date), "w")

create_datasets(f,train_data,x_flat,y_flat,"train")

#====== test files ========

#print("making test h5 file.")

test_out = open("templates/template_events_d99353.out", "r")
##print("writing to file %i \n",i)
lines = test_out.readlines()
test_out.close()

n_test = int((len(lines)-2)/14)
#print("n_test = ",n_test)

#"image" size = 13x21x1
test_data = np.zeros((n_test,13,21,1))
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
##print(test_data[0].reshape((21,13)))
cota,cotb,x_position,y_position = convert_pav_to_cms()
##print(x_position_pav[0],y_position_pav[0])
##print(x_position[0],y_position[0])

#n_elec were scaled down by 10 so multiply
test_data = 10*test_data
#print("multiplied all elements by 10")
##print(test_data[0].reshape((21,13)))

apply_noise(test_data,fe_type)
##print(test_data[0].reshape((21,13)))
test_data = apply_threshold(test_data,threshold)
##print(test_data[0].reshape((21,13)))

center_clusters(test_data)
##print(test_data[0].reshape((21,13)))
##print(x_position[0],y_position[0])

project_matrices_xy(test_data)
##print(x_flat[0],y_flat[0])
##print(clustersize_x[0],clustersize_y[0])

f = h5py.File("h5_files/test_%s_%s.hdf5"%(filename,date), "w")

create_datasets(f,test_data,x_flat,y_flat,"test")



