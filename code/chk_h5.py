import numpy as np
import h5py
from skimage.measure import label
import numpy.random as rng

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

date = "oct19"


f = h5py.File("h5_files/test_d49350_%s.hdf5"%(date), "w")

test_out = open("templates/template_events_d99352_subset.out", "r")
#print("writing to file %i \n",i)
lines = test_out.readlines()
test_out.close()

n_test = int((len(lines)-2)/14)
#n_test=100
print("n_test = ",n_test)
n_per_file = int(len(lines)/14)
#n_per_file=100

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


#delete first 2 lines
pixelsize = lines[1] 		
del lines[0:2]

n=0

for j in range(0,n_per_file):

	#there are n 13x21 arrays in the file, extract each array 
	array2d = [[float(digit) for digit in line.split()] for line in lines[n+1:n+14]]
	#reshape to (13,21,1) -> "image"
	#convert from pixelav sensor coords to normal coords
	testdata = np.array(array2d)
	testdata = np.flip(testdata,0)
	testdata = np.flip(testdata,1)
	#print('flipped')
	#print(test_data[j])
	test_data[j]=testdata[:,:,np.newaxis]

	test_data[j]=test_data[j]*10.
	noise_1 = rng.normal(loc=0.,scale=1.,size=(13*21)).reshape((13,21,1)) #generate a matrix with 21x13 elements from a gaussian dist with mu = 0 and sig = 1
	noise_2 = rng.normal(loc=0.,scale=1.,size=(13*21)).reshape((13,21,1))
	test_data[j]+= gain_frac*noise_1*test_data[j] + readout_noise*noise_2
	below_threshold_i = test_data[j] < 1000
	(test_data[j])[below_threshold_i] = 0
	test_data[j]=(test_data[j]/10.).astype(int)
	#print('added noise and threshold')
	#print(test_data[j])

	#x_flat = test_data[j].reshape((21,13)).sum(axis=1)
	#y_flat = test_data[j].reshape((21,13)).sum(axis=0)

	#clustersize_x[j] = len(np.nonzero(x_flat)[0])
	#clustersize_y[j] = len(np.nonzero(y_flat)[0])

	#preceding each matrix is: x, y, z, cos x, cos y, cos z, nelec
	#cota = cos y/cos z ; cotb = cos x/cos z
	position_data = lines[n].split(' ')
	x_position_pav[j] = float(position_data[1])
	y_position_pav[j] = float(position_data[0])
	cosx[j] = float(position_data[4])
	cosy[j] = float(position_data[3])
	cosz[j] = float(position_data[5])

	pixelsize_data = pixelsize.split('  ')
	pixelsize_x[j] = float(pixelsize_data[1]) 
	pixelsize_y[j] = float(pixelsize_data[0])
	pixelsize_z[j] = float(pixelsize_data[2])

	n+=14

print("read in matrices from txt file\nflipped all matrices")


#============= preprocessing =====================
#switching out of pixelav coords to localx and localy
#remember that h5 files have already been made with transposed matrices

#float z_center = zsize/2.0;
#float xhit = x1 + (z_center - z1) * cosx/cosz; cosx/cosz = cotb
#float yhit = y1 + (z_center - z1) * cosy/cosz; cosy/cosz = cota
#z1 is always 0 

cota = cosx/cosz
cotb = cosy/cosz
x_position = -(x_position_pav + (pixelsize_z/2.)*cota)
y_position = -(y_position_pav + (pixelsize_z/2.)*cotb)

print("transposed all test matrices\nconverted test_labels from pixelav coords to cms coords \ncomputed test cota cotb\n")


#shifting wav of cluster to matrix centre
#for index in np.arange(len(test_data)):
for index in np.arange(len(x_position)):

	#find clusters
	one_mat = test_data[index].reshape((13,21))
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
	clustersize_x[index] = len(np.unique(largest_idxs_x))
	clustersize_y[index] = len(np.unique(largest_idxs_y))

	if(clustersize_x[index]==1):

		print(one_mat)
		print(labels)
		#print('max_cluster_size=',max_cluster_size)
		#print('largest_idxs=',[largest_idxs_x,largest_idxs_y])
		print('clustersize_x=',clustersize_x[index],'clustersize_y=',clustersize_y[index])
		#find geometric centre of the main cluster using avg
		center_x = round(np.mean(largest_idxs_x))
		center_y = round(np.mean(largest_idxs_y))
		print('centers',[center_x,center_y])
		#if the geometric centre is not at (7,11) shift cluster
		nonzero_list = np.asarray(np.nonzero(test_data[index]))
		nonzero_x = nonzero_list[0,:]
		nonzero_y = nonzero_list[1,:]
		if(center_x<6):
			#shift down
			shift = int(6-center_x)
			if(np.amax(nonzero_x)+shift<=12):
				print(x_position[index],y_position[index])
				one_mat=np.roll(one_mat,shift,axis=0)
				x_position[index]-=pixelsize_x[index]*shift
				print('shift down ',shift)
				print(x_position[index],y_position[index])
				print(one_mat)

		if(center_x>6):
			#shift up
			shift = int(center_x-6)
			if(np.amin(nonzero_x)-shift>=0):
				print(x_position[index],y_position[index])
				one_mat=np.roll(one_mat,-shift,axis=0)
				x_position[index]+=pixelsize_x[index]*shift
				print('shift up ',shift)
				print(x_position[index],y_position[index])
				print(one_mat)

		if(center_y<10):
			#shift right
			shift = int(10-center_y)
			if(np.amax(nonzero_y)+shift<=20):
				print(x_position[index],y_position[index])
				one_mat=np.roll(one_mat,shift,axis=1)
				y_position[index]+=pixelsize_y[index]*shift
				print('shift right ',shift)
				print(x_position[index],y_position[index])
				print(one_mat)

		if(center_y>10):
			#shift left
			shift = int(center_y-10)
			if(np.amin(nonzero_y)-shift>=0):
				print(x_position[index],y_position[index])
				one_mat=np.roll(one_mat,-shift,axis=1)
				y_position[index]-=pixelsize_y[index]*shift
				print('shift left ',shift)
				print(x_position[index],y_position[index])
				print(one_mat)

	test_data[index]=one_mat[:,:,np.newaxis]

'''
	nonzero_list = np.transpose(np.asarray(np.nonzero(test_data[index])))
	nonzero_elements = test_data[index][np.nonzero(test_data[index])]
	#print(nonzero_elements.shape)
	nonzero_i = nonzero_list[:,0]-6. #x indices
	#print(nonzero_i.shape)
	nonzero_j = nonzero_list[:,1]-10. #y indices
	shift_i = -int(np.sum(nonzero_i)/len(nonzero_i))
	shift_j = -int(np.sum(nonzero_j)/len(nonzero_j))
	#print(wav_i-10,wav_j-6)

#CHEXK THIS ONWARDS
	if(shift_i>0 and np.amax(nonzero_i)!=12):
		#print(test_data[index].reshape((21,13)))
		#print(x_position[index],y_position[index])
		#print(shift_i,shift_j)
		#shift down iff there is no element at the last column
		test_data[index] = np.roll(test_data[index],shift_i,axis=0)
		#shift hit position too
		x_position[index]-=pixelsize_x[index]*shift_i

		#print(test_data[index].reshape((21,13)))
		#print(x_position[index],y_position[index])
		#print('shift down done')

	if(shift_i<0 and np.amin(nonzero_i)!=0):
		#print(test_data[index].reshape((21,13)))
		#print(x_position[index],y_position[index])
		#print(shift_i,shift_j)

		#shift up iff there is no element at the first column
		test_data[index] = np.roll(test_data[index],shift_i,axis=0)
		#shift hit position too
		x_position[index]-=pixelsize_x[index]*shift_i

		#print(test_data[index].reshape((21,13)))
		#print(x_position[index],y_position[index])
		#print('shift up done')
	if(shift_j>0 and np.amax(nonzero_j)!=20):
		#print(test_data[index].reshape((21,13)))
		#print(x_position[index],y_position[index])
		#print(shift_i,shift_j)
		print(index)
		
		#shift right iff there is no element in the last row
		test_data[index] = np.roll(test_data[index],shift_j,axis=1)
		#shift hit position too
		y_position[index]+=pixelsize_y[index]*shift_j
		

	if(shift_j<0 and np.amin(nonzero_j)!=0):
		#print(test_data[index].reshape((21,13)))
		#print(x_position[index],y_position[index])
		#print(shift_i,shift_j)
		print(index)

		#shift left iff there is no element in the first row
		test_data[index] = np.roll(test_data[index],shift_j,axis=1)
		#shift hit position too
		y_position[index]+=pixelsize_y[index]*shift_j
			

print("shifted wav of clusters to matrix centres")
'''
