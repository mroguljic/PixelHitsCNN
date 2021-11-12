#----------------
# simulate double width pixels 
# Author: Sanjana Sekhar
# Date: 10/29/21
#-----------------
import numpy as np
import numpy.random as rng

def simulate_double_width_1d(x_flat,y_flat,clustersize_x,clustersize_y,x_position,y_position,cota,cotb,n_double):

	# only for 1d 
	# 3 cases: double in x, double in y, double in x and y
	# for 1d, case 3 wont make a difference to how the flat matrices look - bother about this in 2d 
	# simulate n_double matrices for case 1 and 2
	'''
	general algo: 
	skip clsize = 1 and 2
	for N >2, clsize N simulates double width pix of clsize_d = N-1
	this only works for clsize_d <= 12/20 ie if there is a double width pixel in a cluster of size 13 or 20 -> skip it for now
	simulation: 
	clsize = 2: let choice of i only be the first idx x[i]+x[i+1]/2, clsize--;
	clsize = 3: let choice of i only be the first,second idx x[i]+x[i+1]/2, clsize--;
	so on and so forth
	
	for simulating 2 double width next to each other:
	clsize N in single -> simulates clsize N-3 in 2 double
	start with clsize 4
	only a maximum of clsize 10/18 can be simulated with 2 double width

	'''
	#choose clusters whose size is not 1 in x or y
	print("no of available choices: ",len(np.argwhere(clustersize_x>2)[:,0]))	
	if len(np.argwhere(clustersize_x>2)[:,0]) < n_double: double_idx_x = np.argwhere(clustersize_x>2)[:,0]
	else: double_idx_x = rng.choice(np.argwhere(clustersize_x>2)[:,0],size=n_double,replace=False)
	if len(np.argwhere(clustersize_y>2)[:,0]) < n_double: double_idx_y = np.argwhere(clustersize_y>2)[:,0]
	else: double_idx_y = rng.choice(np.argwhere(clustersize_y>2)[:,0],size=n_double,replace=False)
	#print("no of available choices: ",np.intersect1d(np.argwhere(clustersize_x!=1)[:,0],np.argwhere(clustersize_x!=2)[:,0]),np.intersect1d(np.argwhere(clustersize_y!=1)[:,0],np.argwhere(clustersize_y!=2)[:,0]))
	print("old x_flat shape = ",x_flat.shape,"old y_flat shape = ",y_flat.shape)
	
	flat_list,clustersize_list,pos_list,cota_list,cotb_list = [],[],[],[],[]
	count=0

	for i in double_idx_x:
		# for x matrices
		#if clustersize_x[i]==1: print(x_flat[i])
		nonzero_idx = np.array(np.nonzero(x_flat[i])).reshape((int(clustersize_x[i]),))
		#clustersize_x[i]-=1 
		#since this now simulates a double col pix, the cluster size goes down by 1
		#if clustersize_x[i][0]-1>5: n_choices = 5
		n_choices = clustersize_x[i][0]+3

		#simulate all configs of double width
		for j in rng.choice(nonzero_idx[:-1],size=int(n_choices),replace=True):
			one_mat = np.copy(x_flat[i])
			#if count < 30:
			#	print("cluster x")
			#	print(one_mat)
			one_mat[j] = one_mat[j+1] = (one_mat[j]+one_mat[j+1])/2.
			flat_list.append(one_mat.tolist())
			clustersize_list.append((clustersize_x[i]-1).tolist())
			pos_list.append(x_position[i].tolist())
			cota_list.append(cota[i].tolist())
			cotb_list.append(cotb[i].tolist())
			count+=1
			#if count < 30:
			#	print("1 dpix cluster x")
			#	print(one_mat)

			# ==================== simulate 2 dpix =====================================
			if j<nonzero_idx[-3] and clustersize_x[i][0] > 3:
				one_mat[j+2] = one_mat[j+3] = (one_mat[j+2]+one_mat[j+3])/2.
				flat_list.append(one_mat.tolist())
				clustersize_list.append((clustersize_x[i]-3).tolist())
				pos_list.append(x_position[i].tolist())
				cota_list.append(cota[i].tolist())
				cotb_list.append(cotb[i].tolist())
				count+=1

			#	if count < 30:
			#		print("2 dpix cluster x")
			#		print(one_mat)


	x_flat = np.vstack((x_flat,np.array(flat_list).reshape((count,13))))
	clustersize_x = np.vstack((clustersize_x,np.array(clustersize_list).reshape((count,1))))
	x_position = np.vstack((x_position,np.array(pos_list).reshape((count,1))))
	cota_x = np.vstack((cota,np.array(cota_list).reshape((count,1))))
	cotb_x = np.vstack((cotb,np.array(cotb_list).reshape((count,1))))

	flat_list,clustersize_list,pos_list,cota_list,cotb_list = [],[],[],[],[]
	count=0

	for i in double_idx_y:
		# for y matrices
		#if clustersize_y[i]==1: print(y_flat[i])
		nonzero_idx = np.array(np.nonzero(y_flat[i])).reshape((int(clustersize_y[i]),))
		#clustersize_y[i]-=1 
		#since this now simulates a double col pix, the cluster size goes down by 1
		if clustersize_y[i][0]-1>7: n_choices = 4
		else: n_choices = clustersize_y[i][0]-1

		#simulate all configs of double width
		for j in rng.choice(nonzero_idx[:-1],size=int(n_choices),replace=False):
			one_mat = np.copy(y_flat[i])
			#if count < 30:
			#	print("cluster y")
			#	print(one_mat)
			one_mat[j] = one_mat[j+1] = (one_mat[j]+one_mat[j+1])/2.
			flat_list.append(one_mat.tolist())
			clustersize_list.append((clustersize_y[i]-1).tolist())
			pos_list.append(y_position[i].tolist())
			cota_list.append(cota[i].tolist())
			cotb_list.append(cotb[i].tolist())
			count+=1
			#if count < 30:
			#	print("1 dpix cluster y")
			#	print(one_mat)
			# ==================== simulate 2 dpix =====================================
			if j<nonzero_idx[-3] and clustersize_y[i][0] > 3:
				one_mat[j+2] = one_mat[j+3] = (one_mat[j+2]+one_mat[j+3])/2.
				flat_list.append(one_mat.tolist())
				clustersize_list.append((clustersize_y[i]-3).tolist())
				pos_list.append(y_position[i].tolist())
				cota_list.append(cota[i].tolist())
				cotb_list.append(cotb[i].tolist())
				count+=1

			#	if count < 30:
			#		print("2 dpix cluster y")
			#		print(one_mat)

	y_flat = np.vstack((y_flat,np.array(flat_list).reshape((count,21))))
	clustersize_y = np.vstack((clustersize_y,np.array(clustersize_list).reshape((count,1))))
	y_position = np.vstack((y_position,np.array(pos_list).reshape((count,1))))
	cota_y = np.vstack((cota,np.array(cota_list).reshape((count,1))))
	cotb_y = np.vstack((cotb,np.array(cotb_list).reshape((count,1))))


	print("new x_flat shape = ",x_flat.shape,"new y_flat shape = ",y_flat.shape)
	print("simulated 1 and 2 double width pix in x and y for 1D")

	return x_flat,y_flat,clustersize_x,clustersize_y,x_position,y_position,cota_x,cotb_x,cota_y,cotb_y

def simulate_double_width_2d(cluster_matrices,clustersize_x,clustersize_y,x_position,y_position,cota,cotb,n_double):

	'''
	only for 2d
	8 cases:
	1 in x - DONE
	1 in y - DONE
	1 in x 1 in y - DONE
	2 in x - DONE
	2 in y - DONE
	2 in x 2 in y - DONE
	1 in x 2 in y - DONE
	2 in x 1 in y - DONE
	'''
	temp_idx_x = np.argwhere(clustersize_x>2)[:,0]
	temp_idx_y = np.argwhere(clustersize_y>2)[:,0]

	temp_idx_x2 = np.argwhere(clustersize_x>3)[:,0]
	temp_idx_y2 = np.argwhere(clustersize_y>3)[:,0]

	double_idx_x = np.argwhere(clustersize_x>2)[:,0]
	double_idx_y = rng.choice(np.argwhere(clustersize_y>2)[:,0],size=len(double_idx_x*2),replace=False)
	
	print("no of choices in x = ",len(double_idx_x))
	print("no of choices in y = ",len(double_idx_y))
	

	flat_list,clustersize_list,pos_list,cota_list,cotb_list = [],[],[],[],[]
	count=0

	for i in double_idx_x:
		# simulate 1 in x	
		one_mat = np.copy(cluster_matrices[i]).reshape((13,21))
		nonzero_idx = np.unique(np.array(np.nonzero(one_mat))[0]) #choose 1 x double col 
		n_choices = 2
		for j in rng.choice(nonzero_idx[:-1],size=int(n_choices),replace=False):
		
			one_mat[j]=one_mat[j+1]=(one_mat[j]+one_mat[j+1])/2
			flat_list.append(one_mat.flatten().tolist())
			clustersize_list.append((clustersize_x[i]-1).tolist())
			pos_list.append(x_position[i].tolist())
			cota_list.append(cota[i].tolist())
			cotb_list.append(cotb[i].tolist())
			count+=1

			# simulate 2 in x	
			if j<nonzero_idx[-3] and clustersize_x[i][0] > 3:
				one_mat[j+2] = one_mat[j+3] = (one_mat[j+2]+one_mat[j+3])/2.
				flat_list.append(one_mat.flatten().tolist())
				clustersize_list.append((clustersize_x[i]-3).tolist())
				pos_list.append(x_position[i].tolist())
				cota_list.append(cota[i].tolist())
				cotb_list.append(cotb[i].tolist())
				count+=1

	cluster_matrices_x = np.vstack((cluster_matrices,np.array(flat_list).reshape((count,13,21,1))))
	clustersize_x = np.vstack((clustersize_x,np.array(clustersize_list).reshape((count,1))))
	x_position = np.vstack((x_position,np.array(pos_list).reshape((count,1))))
	cota_x = np.vstack((cota,np.array(cota_list).reshape((count,1))))
	cotb_x = np.vstack((cotb,np.array(cotb_list).reshape((count,1))))

	flat_list,clustersize_list,pos_list,cota_list,cotb_list = [],[],[],[],[]
	count=0

	for i in double_idx_y:
		# simulate 1 in y	
		one_mat = np.copy(cluster_matrices[i]).reshape((13,21))
		nonzero_idx = np.unique(np.array(np.nonzero(one_mat))[1]) #choose 1 y double col 
		n_choices = 2
		for j in rng.choice(nonzero_idx[:-1],size=int(n_choices),replace=False):
			#if(count<30): print(one_mat)	
			one_mat[:,j]=one_mat[:,j+1]=(one_mat[:,j]+one_mat[:,j+1])/2
			#if(count<30): print(one_mat)
			flat_list.append(one_mat.flatten().tolist())
			clustersize_list.append((clustersize_y[i]-1).tolist())
			pos_list.append(y_position[i].tolist())
			cota_list.append(cota[i].tolist())
			cotb_list.append(cotb[i].tolist())
			count+=1

			# simulate 2 in y	
			if j<nonzero_idx[-3] and clustersize_y[i][0] > 3:
				one_mat[:,j+2] = one_mat[:,j+3] = (one_mat[:,j+2]+one_mat[:,j+3])/2.
				#if(count<30): print(one_mat)
				flat_list.append(one_mat.flatten().tolist())
				clustersize_list.append((clustersize_y[i]-3).tolist())
				pos_list.append(y_position[i].tolist())
				cota_list.append(cota[i].tolist())
				cotb_list.append(cotb[i].tolist())
				count+=1

	cluster_matrices_y = np.vstack((cluster_matrices,np.array(flat_list).reshape((count,13,21,1))))
	clustersize_y = np.vstack((clustersize_y,np.array(clustersize_list).reshape((count,1))))
	y_position = np.vstack((y_position,np.array(pos_list).reshape((count,1))))
	cota_y = np.vstack((cota,np.array(cota_list).reshape((count,1))))
	cotb_y = np.vstack((cotb,np.array(cotb_list).reshape((count,1))))

	print("simulated 1 and 2 double width pix in x and y for 2D")
	
	flat_list,clustersize_x_list,pos_x_list,clustersize_y_list,pos_y_list,cota_list,cotb_list = [],[],[],[],[],[],[]
	count=0
	double_idx_xy = np.intersect1d(temp_idx_x,temp_idx_y)
	print("no of choices in 1x1y = ",len(double_idx_xy))

	for i in double_idx_xy:
		# simulate 1 in x 1 in y
		one_mat = np.copy(cluster_matrices[i]).reshape((13,21))
		nonzero_idx_x = np.unique(np.array(np.nonzero(one_mat))[0]) 
		nonzero_idx_y = np.unique(np.array(np.nonzero(one_mat))[1]) 
		n_choices = 1
		j1 = rng.choice(nonzero_idx_x[:-1],size=int(n_choices),replace=False)
		j2 = rng.choice(nonzero_idx_y[:-1],size=int(n_choices),replace=False)
		one_mat[j1]=one_mat[j1+1]=(one_mat[j1]+one_mat[j1+1])/2
		one_mat[:,j2]=one_mat[:,j2+1]=(one_mat[:,j2]+one_mat[:,j2+1])/2
		flat_list.append(one_mat.flatten().tolist())
		clustersize_x_list.append((clustersize_x[i]-1).tolist())
		clustersize_y_list.append((clustersize_y[i]-1).tolist())
		pos_x_list.append(x_position[i].tolist())
		pos_y_list.append(y_position[i].tolist())
		cota_list.append(cota[i].tolist())
		cotb_list.append(cotb[i].tolist())
		count+=1

	print("simulated 1 in x + 1 in y double width pix for 2D")

	double_idx_xy = np.intersect1d(temp_idx_x,temp_idx_y2)
	print("no of choices in 1x2y = ",len(double_idx_xy))

	for i in double_idx_xy:
		# simulate 1 in x 2 in y
		one_mat = np.copy(cluster_matrices[i]).reshape((13,21))
		nonzero_idx_x = np.unique(np.array(np.nonzero(one_mat))[0]) 
		nonzero_idx_y = np.unique(np.array(np.nonzero(one_mat))[1]) 
		n_choices_x = 2
                if clustersize_y[i][0]>4: n_choices_y = 2
                else: n_choices_y = 1
		for j1 in rng.choice(nonzero_idx_x[:-1],size=int(n_choices),replace=False):
			for j2 in rng.choice(nonzero_idx_y[:-3],size=int(n_choices),replace=False):
				one_mat[j1]=one_mat[j1+1]=(one_mat[j1]+one_mat[j1+1])/2
				one_mat[:,j2]=one_mat[:,j2+1]=(one_mat[:,j2]+one_mat[:,j2+1])/2
				one_mat[:,j2+2]=one_mat[:,j2+3]=(one_mat[:,j2+2]+one_mat[:,j2+3])/2
				flat_list.append(one_mat.flatten().tolist())
				clustersize_x_list.append((clustersize_x[i]-1).tolist())
				clustersize_y_list.append((clustersize_y[i]-3).tolist())
				pos_x_list.append(x_position[i].tolist())
				pos_y_list.append(y_position[i].tolist())
				cota_list.append(cota[i].tolist())
				cotb_list.append(cotb[i].tolist())
				count+=1

	print("simulated 1 in x + 2 in y double width pix for 2D")

	double_idx_xy = np.intersect1d(temp_idx_x2,temp_idx_y)
	print("no of choices in 2x1y = ",len(double_idx_xy))

	for i in double_idx_xy:

		# simulate 2 in x 1 in y
		one_mat = np.copy(cluster_matrices[i]).reshape((13,21))
		nonzero_idx_x = np.unique(np.array(np.nonzero(one_mat))[0]) 
		nonzero_idx_y = np.unique(np.array(np.nonzero(one_mat))[1]) 
		if clustersize_x[i][0]>4: n_choices_x = 2
                else: n_choices_x = 1
                n_choices_y = 1
		for j1 in rng.choice(nonzero_idx_x[:-3],size=int(n_choices),replace=False):
			for j2 in rng.choice(nonzero_idx_y[:-1],size=int(n_choices),replace=False):
				one_mat[j1]=one_mat[j1+1]=(one_mat[j1]+one_mat[j1+1])/2
				one_mat[j1+2]=one_mat[j1+3]=(one_mat[j1+2]+one_mat[j1+3])/2
				one_mat[:,j2]=one_mat[:,j2+1]=(one_mat[:,j2]+one_mat[:,j2+1])/2
				flat_list.append(one_mat.flatten().tolist())
				clustersize_x_list.append((clustersize_x[i]-3).tolist())
				clustersize_y_list.append((clustersize_y[i]-1).tolist())
				pos_x_list.append(x_position[i].tolist())
				pos_y_list.append(y_position[i].tolist())
				cota_list.append(cota[i].tolist())
				cotb_list.append(cotb[i].tolist())
				count+=1

	print("simulated 2 in x + 1 in y double width pix for 2D")

	double_idx_xy = np.intersect1d(temp_idx_x2,temp_idx_y2)
	print("no of choices in 2x2y = ",len(double_idx_xy))

	for i in double_idx_xy:

		# simulate 2 in x 2 in y
		one_mat = np.copy(cluster_matrices[i]).reshape((13,21))
		nonzero_idx_x = np.unique(np.array(np.nonzero(one_mat))[0]) 
		nonzero_idx_y = np.unique(np.array(np.nonzero(one_mat))[1]) 
		if clustersize_x[i][0]>4: n_choices_x = 2
		else: n_choices_x = 1
		if clustersize_y[i][0]>4: n_choices_y = 2
		else: n_choices_y = 1
		for j1 in rng.choice(nonzero_idx_x[:-3],size=int(n_choices_x),replace=False):
			for j2 in rng.choice(nonzero_idx_y[:-3],size=int(n_choices_y),replace=False):
				one_mat[j1]=one_mat[j1+1]=(one_mat[j1]+one_mat[j1+1])/2
				one_mat[j1+2]=one_mat[j1+3]=(one_mat[j1+2]+one_mat[j1+3])/2
				one_mat[:,j2]=one_mat[:,j2+1]=(one_mat[:,j2]+one_mat[:,j2+1])/2
				one_mat[:,j2+2]=one_mat[:,j2+3]=(one_mat[:,j2+2]+one_mat[:,j2+3])/2
				flat_list.append(one_mat.flatten().tolist())
				clustersize_x_list.append((clustersize_x[i]-3).tolist())
				clustersize_y_list.append((clustersize_y[i]-3).tolist())
				pos_x_list.append(x_position[i].tolist())
				pos_y_list.append(y_position[i].tolist())
				cota_list.append(cota[i].tolist())
				cotb_list.append(cotb[i].tolist())
				count+=1

	cluster_matrices_x = np.vstack((cluster_matrices_x,np.array(flat_list).reshape((count,13,21,1))))
	cluster_matrices_y = np.vstack((cluster_matrices_y,np.array(flat_list).reshape((count,13,21,1))))
	clustersize_x = np.vstack((clustersize_x,np.array(clustersize_x_list).reshape((count,1))))
	x_position = np.vstack((x_position,np.array(pos_x_list).reshape((count,1))))
	cota_x = np.vstack((cota_x,np.array(cota_list).reshape((count,1))))
	cotb_x = np.vstack((cotb_x,np.array(cotb_list).reshape((count,1))))
	clustersize_y = np.vstack((clustersize_y,np.array(clustersize_y_list).reshape((count,1))))
	y_position = np.vstack((y_position,np.array(pos_y_list).reshape((count,1))))
	cota_y = np.vstack((cota_y,np.array(cota_list).reshape((count,1))))
	cotb_y = np.vstack((cotb_y,np.array(cotb_list).reshape((count,1))))

	print("simulated 2 in x + 2 in y double width pix for 2D")

	print("total no of 2d x clusters = ",len(cluster_matrices_x)," total no of 2d x clusters = ",len(cluster_matrices_y))
	return cluster_matrices_x,cluster_matrices_y, clustersize_x,clustersize_y,x_position,y_position,cota_x,cotb_x,cota_y,cotb_y
