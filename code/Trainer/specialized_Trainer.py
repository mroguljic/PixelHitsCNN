import json
import h5py
import numpy as np
import time
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import load_model
from tensorflow import exp
from tensorflow import device
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Add
from tensorflow import clip_by_value
import tensorflow as tf
from models import losses
from models import architectures
import cmsml
import psutil
import plotting
from models import AlphaScheduler 
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class Trainer:
    #Class to load .json file and launch trainings

    def __init__(self,json_file,layer,axis):
        #axis: "x" or "y"
        assert axis=="x" or axis=="y", f"Invalid axis '{axis}'. Valid choices are 'x' or 'y'"
        self.axis = axis
        self.testing_input_flag = False
        self.training_input_flag = False
        self.layer = layer
        self.resolution = -999.
        self.bias = -999.
        
        with open(json_file) as f:
            config = json.load(f)[layer]
            self.epochs = config["epochs"]
            self.batch_size = config["batch_size"]
            self.loss_name = config["loss_name"]
            self.modelname = config["modelname"]
            if axis=="x":
                self.train_h5 = config["train_x_h5"]
                self.test_h5 = config["test_x_h5"]
                self.checkpoint = config["checkpoint_x"]
                self.model_dest = config["model_dest_x"]
                self.pitch      = config["pitch_x"]
            else:
                self.train_h5 = config["train_y_h5"]
                self.test_h5 = config["test_y_h5"]
                self.checkpoint = config["checkpoint_y"]
                self.model_dest = config["model_dest_y"]
                self.pitch      = config["pitch_y"]

    def print_h5_keys(self, h5file):
        def visit_fn(name, obj):
            if isinstance(obj, h5py.Group) or isinstance(obj, h5py.Dataset):
                print(name)
        h5file.visititems(visit_fn)

    
    def prepare_training_input(self):
        n_clusters = -1#Set to -1 for all or some number like 100000 for faster debugging
        print(f"Loading training clusters from {self.train_h5}")
        f = h5py.File(self.train_h5, 'r')
        print("-----Keys-----")
        self.print_h5_keys(f)
        print("-----Keys-----")
        if f'train_{self.axis}_flat' in f:#In some files, train_ prefix is not present
            pix_flat_train = f[f'train_{self.axis}_flat'][0:n_clusters]
        else:
            pix_flat_train = f[f'{self.axis}_flat'][0:n_clusters]
        cota_train = f['cota'][0:n_clusters]
        cotb_train = f['cotb'][0:n_clusters]
        position_train = f[self.axis][0:n_clusters] 
        clustersize_train = f["clustersize"][0:n_clusters]
        clucharges_train = f["cluster_charge"][0:n_clusters]
        f.close()
        print(f"Loaded {len(pix_flat_train)} clusters")
        perm = np.arange(len(pix_flat_train)) 
        np.random.shuffle(perm)
        
        pix_flat_train = pix_flat_train[perm]
        cota_train = cota_train[perm]
        cotb_train = cotb_train[perm]
        position_train = position_train[perm]
        clustersize_train = clustersize_train[perm]
        clucharges_train = clucharges_train[perm]
        angles_train = np.hstack((cota_train,cotb_train))

        self.pixels_train = pix_flat_train
        self.position_train = position_train
        self.angles_train = angles_train
        self.clustersize_train = clustersize_train
        self.clucharges_train = clucharges_train
        self.training_input_flag = True
        print("Using {:.0f} MB of memory".format(psutil.Process().memory_info().rss / 1024 ** 2))

    def prepare_testing_input(self):
        n_clusters = -1#Set to -1 for all or some number like 100000 for faster debugging

        print(f"Loading testing clusters from {self.test_h5}")
        f = h5py.File(self.test_h5, 'r')
        print("-----Keys-----")
        self.print_h5_keys(f)
        print("-----Keys-----")
        if f'test_{self.axis}_flat' in f:#In some files, test_ prefix is not present
            pix_flat_test = f[f'test_{self.axis}_flat'][0:n_clusters]
        else:
            pix_flat_test = f[f'{self.axis}_flat'][0:n_clusters]
        cota_test = f['cota'][0:n_clusters]
        cotb_test = f['cotb'][0:n_clusters]
        position_test = f[self.axis][0:n_clusters] 
        clustersize_test = f["clustersize"][0:n_clusters]
        clucharges_test = f["cluster_charge"][0:n_clusters]
        angles_test = np.hstack((cota_test,cotb_test))
        f.close()

        self.pixels_test = pix_flat_test
        self.position_test = position_test
        self.angles_test = angles_test
        self.clustersize_test = clustersize_test
        self.clucharges_test = clucharges_test
        print(np.shape(self.pixels_test))
        print(np.shape(self.position_test))
        print(np.shape(self.angles_test))
        print(np.shape(self.clustersize_test))
        print(np.shape(self.clucharges_test))
        print("Using {:.0f} MB of memory".format(psutil.Process().memory_info().rss / 1024 ** 2))
        self.testing_input_flag = True

    def train(self):
        if not self.training_input_flag:
            self.prepare_training_input()
        optimizer = Adam()
        validation_split = 0.02
        dropout_level = 0.10

        train_time = time.process_time()

        if self.axis == "x":
            inputs = Input(shape=(13, 1), name="pixel_projection_x")  # 13 in x dimension
            input_dim = 16
        else:
            inputs = Input(shape=(21, 1), name="pixel_projection_y")  # 21 in y dimension
            input_dim = 24

        angles = Input(shape=(2,), name="angles")
        charges = Input(shape=(1,), name="cluster_charge")
        model_fn = getattr(architectures, self.modelname)
        model = model_fn(inputs, angles, charges, input_dim)
        #model.alpha = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        # Display model summary
        model.summary()

        # Compile the model
        run_eagerly = False
        loss_func = getattr(losses, self.loss_name)
        #loss_func = lambda y_true, y_pred: losses.nll_with_annealing(y_true, y_pred, model.alpha)
        model.compile(loss=loss_func, optimizer=optimizer, run_eagerly=run_eagerly, metrics=[loss_func,losses.mse_position])

        #Load weights from checkpoint if exists
        checkpoint_filepath=self.checkpoint
        """if os.path.exists(checkpoint_filepath):
            #print("Skipping loading weights")
            print(f"Loading weights from {checkpoint_filepath}")
            model.load_weights(checkpoint_filepath)"""
            
        #Uncomment if you want to save weights in a different file!
        #checkpoint_filepath = checkpoint_filepath.replace(".ckpt",".{epoch:02d}-{val_loss:.2f}.ckpt")
        
        #decay_rate = 1.0/(self.epochs-3)#Stay at least two epochs on alpha=0.0 
        #print(f"Alpha decay rate {decay_rate:.3f}")
        #alpha_scheduler = AlphaScheduler.AlphaScheduler(total_epochs=self.epochs, initial_alpha=1.0, decay_rate=decay_rate)
        callbacks = [
        ModelCheckpoint(filepath=checkpoint_filepath,
                        save_best_only=True,
                        save_weights_only=True,
                        #monitor='val_loss',
                        monitor=self.loss_name,
                        save_freq="epoch")
                        #save_freq=1000)
                        #alpha_scheduler
        ]

        # Fit data to model
        history = model.fit([self.pixels_train[:,:,np.newaxis],self.angles_train,self.clucharges_train], [self.position_train],
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        callbacks=callbacks,
                        validation_split=validation_split)


        model.save(self.model_dest)
        cmsml.tensorflow.save_graph(self.model_dest+".pb", model, variables_to_constants=True)
        cmsml.tensorflow.save_graph(self.model_dest+".pb.txt", model, variables_to_constants=True)

        if not os.path.exists("plots"):
            os.makedirs("plots")
        history_plot = f"plots/{self.layer}_{self.axis}_history.png"
        plotting.plot_dnn_loss(history.history,history_plot)
        plotting.plot_nll_and_mse(history.history, f"plots/{self.layer}_{self.axis}")
        print("Training time: {:.0f}s".format(time.process_time()-train_time))

    def test(self):
        if not self.testing_input_flag:
            self.prepare_testing_input()
        model = load_model(self.model_dest, custom_objects={self.loss_name: getattr(losses,self.loss_name),"mse_position":losses.mse_position,"mean_pulls":losses.mean_pulls})

        start = time.process_time()
        #with device('/CPU:0'):
        self.pred = model.predict([self.pixels_test[:,:,np.newaxis],self.angles_test,self.clucharges_test], batch_size=400000)
        inference_time_x = time.process_time() - start
        print("Inference on {} cluster took {:.3f} s".format(len(self.pred),inference_time_x))
        residuals = self.pred[:,0] - self.position_test[:,0]
        pulls     = residuals/self.pred[:,1]

        self.resolution = np.std(residuals)
        self.bias       = np.mean(residuals)
        print("Residuals mean and std (microns): {:.3f} +/- {:.3f}".format(self.bias,self.resolution))
        print("Pulls mean and std: {:.3f} +/- {:.3f}".format(np.mean(pulls),np.std(pulls)))

        residuals_output_file = f"plots/{self.layer}_{self.axis}_residuals.pdf"
        pulls_output_file = f"plots/{self.layer}_{self.axis}_pulls.pdf"
        plot_name = f"{self.layer}_{self.axis}"
        plotting.plot_residuals(residuals,residuals_output_file,plot_type="Residuals",name=plot_name)
        plotting.plot_residuals(pulls,pulls_output_file,plot_type="Pulls",name=plot_name)
        plotting.plot_uncertainties(self.pred[:, 1], f"plots/{self.layer}_{self.axis}_uncertainties.pdf")

    def visualize(self):
        if not self.testing_input_flag:
            self.prepare_testing_input()
        model = load_model(self.model_dest, custom_objects={self.loss_name: getattr(losses,self.loss_name),"mse_position":losses.mse_position,"mean_pulls":losses.mean_pulls})
        n_to_plot = 25
        clusters_for_plotting = self.pixels_test[:n_to_plot]
        angles_for_plotting   = self.angles_test[:n_to_plot]
        position_for_plotting = self.position_test[:n_to_plot,0]
        charges_for_plotting = self.clucharges_test[:n_to_plot,0]
        pred = model.predict([clusters_for_plotting[:,:,np.newaxis],angles_for_plotting,charges_for_plotting])
        
        plotting_data_sets = []
        plotting_file_name = f"plots/{self.layer}_{self.axis}.pdf"
        for i in range(n_to_plot):
            temp_data_set = {
            'cluster': clusters_for_plotting[i],
            'angles': angles_for_plotting[i],
            'prediction_uncertainty': pred[i],
            'position': position_for_plotting[i],
            'pixel_pitch': self.pitch,
            'resolution': self.resolution,
            'bias': self.bias

            }
            plotting_data_sets.append(temp_data_set)
        
    
        plotting.plot_clusters(plotting_data_sets,plotting_file_name)

    #plots random sample of clusters within provided range of uncertainty 
    def visualize_cluster_uncertainty(self): 
        if not self.testing_input_flag:
            self.prepare_testing_input()
        model = load_model(self.model_dest, custom_objects={self.loss_name: getattr(losses,self.loss_name),"mse_position":losses.mse_position,"mean_pulls":losses.mean_pulls})
        n_to_plot = 20
        uncertainty_min = 30
        uncertainty_max = 50
        cluster_idx = []
        for x in range(self.pred.shape[0]):
            if self.pred[x,1] >= uncertainty_min and self.pred[x,1] <= uncertainty_max:
                cluster_idx.append(x)
        idx_to_plot = random.sample(cluster_idx, n_to_plot)
        
        clusters_for_plotting = self.pixels_test[idx_to_plot]
        angles_for_plotting   = self.angles_test[idx_to_plot]
        position_for_plotting = self.position_test[idx_to_plot,0]
        charges_for_plotting = self.clucharges_test[idx_to_plot,0]
        pred = model.predict([clusters_for_plotting[:,:,np.newaxis],angles_for_plotting,charges_for_plotting])
        
        plotting_data_sets = []
        plotting_file_name = f"plots/{self.layer}_{self.axis}_{uncertainty_min}_{uncertainty_max}.pdf"
        for i in range(n_to_plot):
            temp_data_set = {
            'cluster': clusters_for_plotting[i],
            'angles': angles_for_plotting[i],
            'prediction_uncertainty': pred[i],
            'position': position_for_plotting[i],
            'pixel_pitch': self.pitch,
            'resolution': self.resolution,
            'bias': self.bias

            }
            plotting_data_sets.append(temp_data_set)
        
    
        plotting.plot_clusters(plotting_data_sets,plotting_file_name)
    
    #Generates a plot of desired data type vs uncertainty
    def plot_vs_uncertainty(self, data_type):
        if not self.testing_input_flag:
            self.prepare_testing_input()
        model = load_model(self.model_dest, custom_objects={self.loss_name: getattr(losses,self.loss_name),"mse_position":losses.mse_position,"mean_pulls":losses.mean_pulls})
        file_name = f"plots/{self.layer}_{self.axis}_{data_type}.png"
        
        uncertainty = self.pred[:,1]
        if data_type == "cot_a":
            data_to_plot = self.angles_test[:,0]
        elif data_type == "cot_b":
            data_to_plot = self.angles_test[:,1]
        elif data_type == "hit_pos":
            data_to_plot = self.position_test[:,0]
        elif data_type == "charge":
            data_to_plot = self.clucharges_test[:,0]
        elif data_type == "size":
            data_to_plot = self.clustersize_test[:,0]
        
        x_min = data_to_plot.min()
        x_max = data_to_plot.max()
        range  = x_max - x_min
        num_bins = range #set to 200 for every other type of data
        x_bins = np.arange(x_min,x_max, range / num_bins)
        temp_bins = np.arange(0,120.6,0.6)
        y_bins = temp_bins[(temp_bins < 115) | (temp_bins >= 120)]
        plt.hist2d(data_to_plot, uncertainty, bins=[x_bins, y_bins], cmap='viridis', norm = LogNorm())
        plt.colorbar(label='Log-scaled count')
        plt.xlabel(data_type)
        plt.ylabel("Uncertainty")
        plt.title(self.layer+" "+self.axis+" "+data_type)
        plt.tight_layout()
        plt.savefig(file_name)
        print("Saving "+file_name)
        plt.close()

    def plot_barycenter_vs_hit(self, data_type):
        temp = []
        file_name = f"data/plots/{self.layer}_{self.axis}_{data_type}_specialized.png"
        if data_type == "test":
            pixel_indices = np.arange(self.pixels_test[0].size)
            for x in range(self.pixels_test.shape[0]):
                charges = self.pixels_test[x]
                temp.append(np.sum(pixel_indices * charges) / np.sum(charges))
            hit_pos = self.position_test[:,0]
        elif data_type == "train":
            pixel_indices = np.arange(self.pixels_train[0].size)
            for x in range(self.pixels_train.shape[0]):
                charges = self.pixels_train[x]
                temp.append(np.sum(pixel_indices * charges) / np.sum(charges))
            hit_pos = self.position_train[:,0]
        
        if self.pitch == 100:
            bin_arr = np.linspace(-200,1500,170)
        else:
            bin_arr = np.linspace(300,1000,70)

        barycenter = np.array(temp) * self.pitch
        residuals = barycenter - np.abs(hit_pos)
        
        mean = np.mean(residuals)
        std = np.std(residuals)

        plt.hist(residuals, bins=bin_arr, edgecolor='black')
        plt.title('Barycenter - hit position (max uncertainty) '+self.layer+"_"+self.axis+"_"+data_type)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.savefig(file_name)
        print("Saving plot "+file_name)
        plt.close()

