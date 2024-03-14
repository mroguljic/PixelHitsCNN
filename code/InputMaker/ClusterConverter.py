from optparse import OptionParser
from optparse import OptionGroup
from argparse import ArgumentParser
import numpy as np
import h5py
import numpy.random as rng
#from skimage.measure import label
from scipy.ndimage.measurements import label
import json
from itertools import islice

class ClusterConverter:
    #Class to load .json file and convert .txt cluster files into "realistic" clusters and store them in hdf5 format

    def __init__(self,json_file,dataset):
        
        with open(json_file) as f:
            json_file = json.load(f)
            config = json_file[dataset]
            layer_settings = json_file["layer_settings"]
            for key, value in json_file["common"].items():
                setattr(self, key, value)

            """
            Conversion example for parameters from Morris' templates :
            Layer    HV    Model    Template ID    Threshold    Double pixel threshold    thresh1_noise_frac    common_noise_frac    gain_frac readout_noise
            1U    375V    dj0340i    1074    2600    2600    0.073    0.06    0.25    350
            """
            for key, value in layer_settings["default"].items():
                setattr(self, key, value)

            layer_overwrite = config["layer_settings_overwrite"]
            for key, value in layer_settings[layer_overwrite].items():
                setattr(self, key, value)    

            self.print_attributes()        

    def print_attributes(self):
        print("--------------------")
        print("Created ClusterConverter object with these settings")
        for attribute, value in vars(self).items():
            print(f"{attribute}: {value}")
        print("--------------------")
    
    def text_to_hdf5(self,input_file,output_file):
        '''
        Master method that does the conversion
        Loads config from .json
        Processes pixelav clusters and converts them to realistic CMS clusters
        Saves them in train or test .hd5f file
        '''
        self.total_x_position = None
        self.total_y_position = None
        self.total_cota_x = None
        self.total_cotb_x = None
        self.total_cota_y = None
        self.total_cotb_y = None
        self.total_clustersize_x = None
        self.total_clustersize_y = None
        self.total_x_flat = None
        self.total_y_flat = None

        def set_or_extend_array(new_array, member_name):
            current_array = getattr(self, member_name)
            if current_array is None:
                # If array member is not defined, set it as the new array
                setattr(self, member_name, new_array)
            else:
                # If array member exists, extend it along the first axis
                setattr(self, member_name, np.concatenate((current_array, new_array), axis=0))

        def count_lines(filename):
            with open(filename, 'r') as file:
                line_count = sum(1 for line in file)
            return line_count

        def set_pixelsize(filename):
            with open(filename, 'r') as file:
                # Read the first line
                file.readline()
                # Read the second line
                second_line = file.readline()
                self.pixelsize = second_line

        def convert_cluster_batch(lines):
            #Input "lines" must be stripped of the first two lines containing template information and pixelsize!
            n_train = int(len(lines)/self.lines_per_cluster)
            n_double = int(self.double_frac*n_train)

            #"image" size = 13x21x1
            self.cluster_matrices = np.zeros((n_train,13,21,1))
            self.x_position_pav = np.zeros((n_train,1))
            self.y_position_pav = np.zeros((n_train,1))
            self.cosx = np.zeros((n_train,1))
            self.cosy = np.zeros((n_train,1))
            self.cosz = np.zeros((n_train,1))
            self.pixelsize_x = np.zeros((n_train,1))
            self.pixelsize_y = np.zeros((n_train,1))
            self.pixelsize_z = np.zeros((n_train,1))
            self.clustersize_x = np.zeros((n_train,1))
            self.clustersize_y = np.zeros((n_train,1))

            self.extract_matrices(lines)
            self.convert_pav_to_cms()
            #n_elec were scaled down by 10 so multiply
            self.cluster_matrices *= 10

            self.apply_noise_threshold()
            self.apply_gain()

            self.center_clusters()

            self.x_flat = np.zeros((len(self.cluster_matrices),13))
            self.y_flat = np.zeros((len(self.cluster_matrices),21))
            self.project_matrices_xy()


            if self.simulate_double:
                #Simulate_double creates new clusters in x and y so the cota/b can diverge between x and y
                print("x_flat before")
                print(np.shape(self.x_flat))
                cota_x,cotb_x,cota_y,cotb_y = self.simulate_double_width_1d(self.cota,self.cotb,n_double)
                print("x_flat after")
                print(np.shape(self.x_flat))
                self.cota_x = cota_x
                self.cota_y = cota_y
                self.cotb_x = cotb_x
                self.cotb_y = cotb_y
            else:
                self.cota_x = self.cota
                self.cota_y = self.cota
                self.cotb_x = self.cotb
                self.cotb_y = self.cotb

            set_or_extend_array(self.x_position, "total_x_position")
            set_or_extend_array(self.y_position, "total_y_position")
            set_or_extend_array(self.cota_x, "total_cota_x")
            set_or_extend_array(self.cota_y, "total_cota_y")
            set_or_extend_array(self.cotb_x, "total_cotb_x")
            set_or_extend_array(self.cotb_y, "total_cotb_y")
            set_or_extend_array(self.clustersize_x, "total_clustersize_x")
            set_or_extend_array(self.clustersize_y, "total_clustersize_y")
            set_or_extend_array(self.x_flat, "total_x_flat")
            set_or_extend_array(self.y_flat, "total_y_flat")

    
        set_pixelsize(input_file)
        temp_file = open(input_file, "r")
        temp_file.close()
        batch_size = 300000
        n_lines = count_lines(input_file)
        n_clusters = (n_lines - 2)/self.lines_per_cluster
        n_batches  = int(np.ceil(n_clusters/batch_size))
        with open(input_file, 'r') as file:
            # Skip the first two lines with template and pixelsize info at the beginning of the file
            next(file)
            next(file)
            '''
            Cluster files can be absolute units, close to 10 GB
            We risk running out of memory if we process all at once
            Processing is done in batches of $batch_size clusters
            '''
            batch_idx = 0
            while True:
                print(f"Batch {batch_idx}/{n_batches}")
                batch = list(islice(file, self.lines_per_cluster*batch_size))
                if not batch:
                    break  # End of file reached
                batch_idx+=1
                # Process the batch
                convert_cluster_batch(batch)
                exit()

        replacer_string_x = f"_x_1d.hdf5"
        replacer_string_y = f"_y_1d.hdf5"
        if not self.simulate_double:
            replacer_string_x = "_nodouble"+replacer_string_x 
            replacer_string_y = "_nodouble"+replacer_string_y 

        f_x_name = output_file.replace(".hdf5",replacer_string_x)
        f_y_name = output_file.replace(".hdf5",replacer_string_y)
        f_x = h5py.File(f_x_name, "w")
        f_y = h5py.File(f_y_name, "w")
        self.create_datasets_1d(f_x,f_y)
        f_x.close()
        f_y.close()
        print(f"Saved clusters to: {f_x_name}")
        print(f"Saved clusters to: {f_y_name}")

    def extract_matrices(self,lines):
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

            self.cluster_matrices[j]=one_mat[:,:,np.newaxis]

            #preceding each matrix is: x, y, z, cos x, cos y, cos z, nelec from pixel av
            #cota = cos x/cos z ; cotb = cos y/cos z
            position_data = lines[n].split(' ')
            self.x_position_pav[j] = float(position_data[1])
            self.y_position_pav[j] = float(position_data[0])
            self.cosx[j] = float(position_data[4])
            self.cosy[j] = float(position_data[3])
            self.cosz[j] = float(position_data[5])

            pixelsize_data = self.pixelsize.split('  ')
            self.pixelsize_x[j] = float(pixelsize_data[1]) #flipped on purpose cus matrix has transposed
            self.pixelsize_y[j] = float(pixelsize_data[0])
            self.pixelsize_z[j] = float(pixelsize_data[2])

            n+=14

    def convert_pav_to_cms(self):
        
        #switching out of pixelav coords to localx and localy
        #remember that h5 files have already been made with transposed matrices 
        '''
        float z_center = zsize/2.0;
        float xhit = x1 + (z_center - z1) * cosx/cosz; cosx/cosz = cota
        float yhit = y1 + (z_center - z1) * cosy/cosz; cosy/cosz = cotb
        x -> -y
        y -> -x
        z1 is always 0 
        '''
        self.cota = self.cosx/self.cosz
        self.cotb = self.cosy/self.cosz
        self.x_position = -(self.x_position_pav + (self.pixelsize_z/2.)*self.cota)
        self.y_position = -(self.y_position_pav + (self.pixelsize_z/2.)*self.cotb)

    def apply_gain(self):
        #add 2 types of noise

        if(self.fe_type==2): #linear gain, Phase 2
            for index in np.arange(len(self.cluster_matrices)):
                hits = self.cluster_matrices[index][np.nonzero(self.cluster_matrices[index])]
                noise_1 = rng.normal(loc=0.,scale=1.,size=len(hits)) #generate a matrix with 21x13 elements from a gaussian dist with mu = 0 and sig = 1
                noise_2 = rng.normal(loc=0.,scale=1.,size=len(hits))
                hits+= self.gain_frac*noise_1*hits + self.readout_noise*noise_2
                self.cluster_matrices[index][np.nonzero(self.cluster_matrices[index])]=hits

        elif(self.fe_type==1): #tanh gain, Phase 1
        #NEED TO CHANGE
            for index in np.arange(len(self.cluster_matrices)):
                #one_mat = self.cluster_matrices[index].reshape((13,21))
                #nonzero_idx = np.nonzero(one_mat)
                #hits = one_mat[nonzero_idx]
                hits = self.cluster_matrices[index][np.nonzero(self.cluster_matrices[index])]
                noise_1 = rng.normal(loc=0.,scale=1.,size=len(hits)) #generate a matrix with 21 elements from a gaussian dist with mu = 0 and sig = 1
                noise_2 = rng.normal(loc=0.,scale=1.,size=len(hits))
                '''
                noise_1,noise_2 = [],[]

                for i in range(13):

                    noise_1_t = rng.normal(loc=0.,scale=1.,size=21) #generate a matrix with 21 elements from a gaussian dist with mu = 0 and sig = 1
                    noise_2_t = rng.normal(loc=0.,scale=1.,size=21)
                    noise_1.append(noise_1_t)
                    noise_2.append(noise_2_t)

                noise_1 = np.array(noise_1).reshape((13,21))
                noise_2 = np.array(noise_2).reshape((13,21))

                noise_1 = noise_1[nonzero_idx]
                noise_2 = noise_2[nonzero_idx]
                '''
                
                adc = ((self.p3+self.p2*np.tanh(self.p0*(hits+ self.vcaloffst)/(7.0*self.vcal) - self.p1)).astype(int)).astype(float)
                hits = (((1.+self.gain_frac*noise_1)*(self.vcal*self.gain*(adc-self.ped))).astype(float) - self.vcaloffst + noise_2*self.readout_noise)

                #signal = ((float)((1.+gain_frac*ygauss[i])*(vcal*gain*(adc-ped))) - vcaloffst + zgauss[i]*readout_noise)/qscale
                #https://github.com/SanjanaSekhar/PixelTemplateProduction/blob/master/src/gen_zp_template.cc#L572
                #https://github.com/SanjanaSekhar/PixelTemplateProduction/blob/master/src/gen_zp_template.cc#L610
                noise_3 = rng.normal(loc=0.,scale=1.,size=1)
                qsmear = 1.+noise_3*self.common_noise_frac
                hits*=qsmear
                self.cluster_matrices[index][np.nonzero(self.cluster_matrices[index])]=hits
                #one_mat[nonzero_idx]=hits
                #self.cluster_matrices[index] = one_mat[:,:,np.newaxis]

    def apply_noise_threshold(self):
        #https://github.com/SanjanaSekhar/PixelTemplateProduction/blob/master/src/gen_zp_template.cc#L584-L610
        below_threshold_i = self.cluster_matrices < 200.
        self.cluster_matrices[below_threshold_i] = 0

        for index in np.arange(len(self.cluster_matrices)):

            hits = self.cluster_matrices[index][np.nonzero(self.cluster_matrices[index])]
            noise_1 = rng.normal(loc=0.,scale=1.,size=len(hits)) #generate a matrix with 21 elements from a gaussian dist with mu = 0 and sig = 1
            noise_2 = rng.normal(loc=0.,scale=1.,size=len(hits))
            '''
            one_mat = self.cluster_matrices[index].reshape((13,21))
            nonzero_idx = np.nonzero(one_mat)
            hits = one_mat[nonzero_idx]
            noise_1,noise_2 = [],[]
            for i in range(13):

                noise_1_t = rng.normal(loc=0.,scale=1.,size=21) #generate a matrix with 21x13 elements from a gaussian dist with mu = 0 and sig = 1
                noise_2_t = rng.normal(loc=0.,scale=1.,size=21)
                noise_1.append(noise_1_t)
                noise_2.append(noise_2_t)

            noise_1 = np.array(noise_1).reshape((13,21))
            noise_2 = np.array(noise_2).reshape((13,21))

            noise_1 = noise_1[nonzero_idx]
            noise_2 = noise_2[nonzero_idx]
            '''
            hits+=noise_1*self.noise
            threshold_noisy = self.threshold*(1+noise_2*self.threshold_noise_frac)
            below_threshold_i = hits < threshold_noisy
            hits[below_threshold_i] = 0.
            self.cluster_matrices[index][np.nonzero(self.cluster_matrices[index])]=hits

    def center_clusters(self):
        
        n_train=len(self.cluster_matrices)
        j, n_empty = 0,0
        #cluster_matrices_new=np.zeros((n_train,13,21,1))
        for index in range(0,n_train):

        #   for index in np.arange(10):
        #       print(self.cluster_matrices[index].reshape((13,21)).astype(int))
                #many matrices are zero cus below thresholf
            
            
            
            #find clusters
            one_mat = self.cluster_matrices[index].reshape((13,21))
            one_mat[one_mat<self.threshold] = 0. #https://github.com/SanjanaSekhar/PixelTemplateProduction/blob/master/src/gen_zp_template.cc#L694

            if(np.all(one_mat==0)):
                n_empty+=1
                continue
            #find largest hit (=seed)
            #seed_index = np.argwhere(one_mat==np.amax(one_mat))[0]
            #find connected components 
            labels,n_clusters = label(one_mat.clip(0,1),structure=np.ones((3,3)))
            #if(index==28): print(one_mat, labels)
        
            max_cluster_size=0
            #if there is more than 1 cluster, the one with largest seed is the main one

            if(n_clusters>1):
            #   if index < 50: 
                #   print("index %i : There are %i clusters"%(index,n_clusters))
                #   print(one_mat)
                #   print("Labels = ", labels)
                #   print(seed_index)   
                for i in range(1,n_clusters+1):
                    cluster_idxs_x = np.argwhere(labels==i)[:,0]
                    cluster_idxs_y = np.argwhere(labels==i)[:,1]
                    #if seed_index in np.argwhere(labels==i): 
                    if np.amax(one_mat) in one_mat[labels==i]:
            #           if(index<50): 
                #           print("inside break ",seed_index, "i= ",i)
                #           print(np.argwhere(labels==i))
                        break
                    #cluster_size = len(cluster_idxs_x)
                    #if cluster_size>max_cluster_size:
                        #max_cluster_size = cluster_size
                largest_idxs_x = cluster_idxs_x
                largest_idxs_y = cluster_idxs_y
                '''
                    #if there are 2 clusters of the same size then the largest hit is the main one
                    elif cluster_size==max_cluster_size: #eg. 2 clusters of size 2
                        if(np.amax(one_mat[largest_idxs_x,largest_idxs_y])<np.amax(one_mat[cluster_idxs_x,cluster_idxs_y])):
                            largest_idxs_x = cluster_idxs_x
                            largest_idxs_y = cluster_idxs_y
                '''
            elif(n_clusters==1):
                i = 1
                largest_idxs_x = np.argwhere(labels==1)[:,0]
                largest_idxs_y = np.argwhere(labels==1)[:,1]
            
            #if(index<30): 
            #   print("i = %i"%i)
            #   print("one_mat before deletion")
            #   print(one_mat)
            one_mat[labels!=i] = 0. #delete everything but the main cluster
            #if n_clusters>1 and index<50: 
            #   print("deleting all clusters but that containing the largest seed") 
            #   print(one_mat)
            #if(index<30): 
                
            #   print("one_mat AFTER deletion")
            #   print(one_mat)
            #find clustersize
            self.clustersize_x[j] = int(len(np.unique(largest_idxs_x)))
            self.clustersize_y[j] = int(len(np.unique(largest_idxs_y)))
            self.cota[j]=self.cota[index]#Overwrite cota/b for empty clusters
            self.cotb[j]=self.cotb[index]
            #find geometric centre of the main cluster using avg
            
            center_x = round(np.mean(largest_idxs_x))
            center_y = round(np.mean(largest_idxs_y))
            #if the geometric centre is not at (7,11) shift cluster

            nonzero_list = np.asarray(np.nonzero(one_mat))
            nonzero_x = nonzero_list[0,:]
            nonzero_y = nonzero_list[1,:]
            if(center_x<6):
                #shift down
                shift = int(6-center_x)
                if(np.amax(nonzero_x)+shift<=12):
                    one_mat=np.roll(one_mat,shift,axis=0)
                    self.x_position[j]+=self.pixelsize_x[index]*shift

            if(center_x>6):
                #shift up
                shift = int(center_x-6)
                if(np.amin(nonzero_x)-shift>=0):
                    one_mat=np.roll(one_mat,-shift,axis=0)
                    self.x_position[j]-=self.pixelsize_x[index]*shift

            if(center_y<10):
                #shift right
                shift = int(10-center_y)
                if(np.amax(nonzero_y)+shift<=20):
                    one_mat=np.roll(one_mat,shift,axis=1)
                    self.y_position[j]+=self.pixelsize_y[index]*shift

            if(center_y>10):
                #shift left
                shift = int(center_y-10)
                if(np.amin(nonzero_y)-shift>=0):
                    one_mat=np.roll(one_mat,-shift,axis=1)
                    self.y_position[j]-=self.pixelsize_y[index]*shift

            one_mat = one_mat/25000.
            self.cluster_matrices[j]=one_mat[:,:,np.newaxis]
            j+=1

        if(n_empty!=0): 
            self.cluster_matrices = self.cluster_matrices[:-n_empty]
            self.clustersize_x = self.clustersize_x[:-n_empty]
            self.clustersize_y = self.clustersize_y[:-n_empty]
            self.x_position = self.x_position[:-n_empty]
            self.y_position = self.y_position[:-n_empty]
            self.cota = self.cota[:-n_empty]
            self.cotb = self.cotb[:-n_empty]

    def project_matrices_xy(self):

        #for dnn
        for index in np.arange(len(self.cluster_matrices)):
            self.x_flat[index] = self.cluster_matrices[index].reshape((13,21)).sum(axis=1)
            self.y_flat[index] = self.cluster_matrices[index].reshape((13,21)).sum(axis=0)

        self.cluster_matrices  = False#Release memory

    def create_datasets_1d(self,f_x,f_y):
        #normalize inputs
        '''
        for index in range(len(x_flat)): #currently testing double width in 1d only 

            max_c = x_flat[index].max()
            x_flat[index] = x_flat[index]/max_c     

        for index in range(len(y_flat)):

            max_c = y_flat[index].max()
            y_flat[index] = y_flat[index]/max_c
        '''

        f_x.create_dataset("x", np.shape(self.total_x_position), data=self.total_x_position)
        f_y.create_dataset("y", np.shape(self.total_y_position), data=self.total_y_position)
        f_x.create_dataset("cota", np.shape(self.total_cota_x), data=self.total_cota_x)
        f_x.create_dataset("cotb", np.shape(self.total_cotb_x), data=self.total_cotb_x)
        f_y.create_dataset("cota", np.shape(self.total_cota_y), data=self.total_cota_y)
        f_y.create_dataset("cotb", np.shape(self.total_cotb_y), data=self.total_cotb_y)
        f_x.create_dataset("clustersize_x", np.shape(self.total_clustersize_x), data=self.total_clustersize_x)
        f_y.create_dataset("clustersize_y", np.shape(self.total_clustersize_y), data=self.total_clustersize_y)
        f_x.create_dataset("x_flat", np.shape(self.total_x_flat), data=self.total_x_flat)
        f_y.create_dataset("y_flat", np.shape(self.total_y_flat), data=self.total_y_flat)

    def simulate_double_width_1d(self,cota,cotb,n_double):

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
        print("no of available choices: ",len(np.argwhere(self.clustersize_x>2)[:,0]))   
        if len(np.argwhere(self.clustersize_x>2)[:,0]) < n_double: double_idx_x = np.argwhere(self.clustersize_x>2)[:,0]
        else: double_idx_x = rng.choice(np.argwhere(self.clustersize_x>2)[:,0],size=n_double,replace=False)
        if len(np.argwhere(self.clustersize_y>2)[:,0]) < n_double: double_idx_y = np.argwhere(self.clustersize_y>2)[:,0]
        else: double_idx_y = rng.choice(np.argwhere(self.clustersize_y>2)[:,0],size=n_double,replace=False)
        #print("no of available choices: ",np.intersect1d(np.argwhere(self.clustersize_x!=1)[:,0],np.argwhere(self.clustersize_x!=2)[:,0]),np.intersect1d(np.argwhere(self.clustersize_y!=1)[:,0],np.argwhere(self.clustersize_y!=2)[:,0]))
        print("old x_flat shape = ",self.x_flat.shape,"old y_flat shape = ",self.y_flat.shape)
        
        flat_list,clustersize_list,pos_list,cota_list,cotb_list = [],[],[],[],[]
        count=0

        for i in double_idx_x:
            # for x matrices
            #if self.clustersize_x[i]==1: print(x_flat[i])
            nonzero_idx = np.array(np.nonzero(self.x_flat[i])).reshape((int(self.clustersize_x[i]),))
            #self.clustersize_x[i]-=1 
            #since this now simulates a double col pix, the cluster size goes down by 1
            #if self.clustersize_x[i][0]-1>5: n_choices = 5
            n_choices = self.clustersize_x[i][0]+3

            #simulate all configs of double width
            for j in rng.choice(nonzero_idx[:-1],size=int(n_choices),replace=True):
                one_mat = np.copy(self.x_flat[i])
                #if count < 30:
                #   print("cluster x")
                #   print(one_mat)
                one_mat[j] = one_mat[j+1] = (one_mat[j]+one_mat[j+1])/2.
                flat_list.append(one_mat.tolist())
                clustersize_list.append((self.clustersize_x[i]-1).tolist())
                pos_list.append(self.x_position[i].tolist())
                cota_list.append(cota[i].tolist())
                cotb_list.append(cotb[i].tolist())
                count+=1
                #if count < 30:
                #   print("1 dpix cluster x")
                #   print(one_mat)

                # ==================== simulate 2 dpix =====================================
                if j<nonzero_idx[-3] and self.clustersize_x[i][0] > 3:
                    one_mat[j+2] = one_mat[j+3] = (one_mat[j+2]+one_mat[j+3])/2.
                    flat_list.append(one_mat.tolist())
                    clustersize_list.append((self.clustersize_x[i]-3).tolist())
                    pos_list.append(self.x_position[i].tolist())
                    cota_list.append(cota[i].tolist())
                    cotb_list.append(cotb[i].tolist())
                    count+=1

                #   if count < 30:
                #       print("2 dpix cluster x")
                #       print(one_mat)

        self.x_flat = np.vstack((self.x_flat,np.array(flat_list).reshape((count,13))))
        self.clustersize_x = np.vstack((self.clustersize_x,np.array(clustersize_list).reshape((count,1))))
        self.x_position = np.vstack((self.x_position,np.array(pos_list).reshape((count,1))))
        cota_x = np.vstack((cota,np.array(cota_list).reshape((count,1))))
        cotb_x = np.vstack((cotb,np.array(cotb_list).reshape((count,1))))

        flat_list,clustersize_list,pos_list,cota_list,cotb_list = [],[],[],[],[]
        count=0

        for i in double_idx_y:
            # for y matrices
            #if self.clustersize_y[i]==1: print(y_flat[i])
            nonzero_idx = np.array(np.nonzero(self.y_flat[i])).reshape((int(self.clustersize_y[i]),))
            #self.clustersize_y[i]-=1 
            #since this now simulates a double col pix, the cluster size goes down by 1
            if self.clustersize_y[i][0]-1>7: n_choices = 4
            else: n_choices = self.clustersize_y[i][0]-1

            #simulate all configs of double width
            for j in rng.choice(nonzero_idx[:-1],size=int(n_choices),replace=False):
                one_mat = np.copy(self.y_flat[i])
                #if count < 30:
                #   print("cluster y")
                #   print(one_mat)
                one_mat[j] = one_mat[j+1] = (one_mat[j]+one_mat[j+1])/2.
                flat_list.append(one_mat.tolist())
                clustersize_list.append((self.clustersize_y[i]-1).tolist())
                pos_list.append(self.y_position[i].tolist())
                cota_list.append(cota[i].tolist())
                cotb_list.append(cotb[i].tolist())
                count+=1
                #if count < 30:
                #   print("1 dpix cluster y")
                #   print(one_mat)
                # ==================== simulate 2 dpix =====================================
                if j<nonzero_idx[-3] and self.clustersize_y[i][0] > 3:
                    one_mat[j+2] = one_mat[j+3] = (one_mat[j+2]+one_mat[j+3])/2.
                    flat_list.append(one_mat.tolist())
                    clustersize_list.append((self.clustersize_y[i]-3).tolist())
                    pos_list.append(self.y_position[i].tolist())
                    cota_list.append(cota[i].tolist())
                    cotb_list.append(cotb[i].tolist())
                    count+=1

                #   if count < 30:
                #       print("2 dpix cluster y")
                #       print(one_mat)

        self.y_flat = np.vstack((self.y_flat,np.array(flat_list).reshape((count,21))))
        self.clustersize_y = np.vstack((self.clustersize_y,np.array(clustersize_list).reshape((count,1))))
        self.y_position = np.vstack((self.y_position,np.array(pos_list).reshape((count,1))))
        cota_y = np.vstack((cota,np.array(cota_list).reshape((count,1))))
        cotb_y = np.vstack((cotb,np.array(cotb_list).reshape((count,1))))


        print("new x_flat shape = ",self.x_flat.shape,"new y_flat shape = ",self.y_flat.shape)
        print("simulated 1 and 2 double width pix in x and y for 1D")
        return cota_x,cotb_x,cota_y,cotb_y

if __name__ == "__main__":
    test_object = ClusterConverter("ClusterConverterConfig.json","L1U")
    input_file = "/uscms/home/roguljic/nobackup/NN_CPE/clusters/L1_U_test.out"
    output_file = "L1_U_train.hdf5"
    test_object.text_to_hdf5(input_file,output_file)

