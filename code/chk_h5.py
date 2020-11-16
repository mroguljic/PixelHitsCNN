import numpy as np
import h5py

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
	test_data[j] = np.array(array2d).transpose()[:,:,np.newaxis]

	x_flat = test_data[j].reshape((21,13)).sum(axis=0)
	y_flat = test_data[j].reshape((21,13)).sum(axis=1)

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

#shifting wav of cluster to matrix centre
#for index in np.arange(len(test_data)):
for index in np.arange(len(x_position)):
	if(clustersize_x[index]==1):
		nonzero_list = np.transpose(np.asarray(np.nonzero(test_data[index])))
		nonzero_elements = test_data[index][np.nonzero(test_data[index])]
		#print(nonzero_elements.shape)
		nonzero_i = nonzero_list[:,0]-10. #x indices
		#print(nonzero_i.shape)
		nonzero_j = nonzero_list[:,1]-5. #y indices
		shift_i = -int(round(np.dot(nonzero_i,nonzero_elements)/np.sum(nonzero_elements)))
		shift_j = -int(round(np.dot(nonzero_j,nonzero_elements)/np.sum(nonzero_elements)))
		#print(wav_i-10,wav_j-6)

		if(shift_i>0 and np.amax(nonzero_i)!=20):
			#print(test_data[index].reshape((21,13)))
			#print(x_position[index],y_position[index])
			#print(shift_i,shift_j)
			#shift down iff there is no element at the last column
			test_data[index] = np.roll(test_data[index],shift_i,axis=0)
			#shift hit position too
			y_position[index]-=pixelsize_y[index]*shift_i

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
			y_position[index]-=pixelsize_y[index]*shift_i

			#print(test_data[index].reshape((21,13)))
			#print(x_position[index],y_position[index])
			#print('shift up done')
		if(shift_j>0 and np.amax(nonzero_j)!=12):
			#print(test_data[index].reshape((21,13)))
			#print(x_position[index],y_position[index])
			#print(shift_i,shift_j)
			print(index)
			
			#shift right iff there is no element in the last row
			test_data[index] = np.roll(test_data[index],shift_j,axis=1)
			#shift hit position too
			x_position[index]+=pixelsize_x[index]*shift_j
			

		if(shift_j<0 and np.amin(nonzero_j)!=0):
			#print(test_data[index].reshape((21,13)))
			#print(x_position[index],y_position[index])
			#print(shift_i,shift_j)
			print(index)

			#shift left iff there is no element in the first row
			test_data[index] = np.roll(test_data[index],shift_j,axis=1)
			#shift hit position too
			x_position[index]+=pixelsize_x[index]*shift_j
			

print("shifted wav of clusters to matrix centres")
