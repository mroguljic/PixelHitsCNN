#============================
# Author: Sanjana Sekhar
# Date: 17 Jan 21
#============================
import h5py
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import r2_score
import numpy as np
import time
from plotter import *
from tensorflow.keras.callbacks import EarlyStopping
from argparse import ArgumentParser
import cmsml



parser = ArgumentParser(description='Train 1D CNN on flattened clusters in x to predict x positions')
parser.add_argument("--img_ext",  default="1dcnn_%s_102523", help="file name extension for residuals and pulls")
parser.add_argument("--h5_date", default="102523", help = "date for h5 file name")
parser.add_argument("--h5_ext",  default="p1_2024_by25k_irrad_BPIXL1", help="h5 file name extension")
parser.add_argument("--early_stop",  default=7, type = int, help="early stopping patience (no. of epochs)")
parser.add_argument("--batch_size",  default=1024, type = int, help="batch size for training")
parser.add_argument("--n_epochs",  default=40, type = int, help="no. of epochs to train for")
options = parser.parse_args()

def mse_with_errors(x_true,x_pred):
    x_position = x_pred[:,:1]
    x_log_error = x_pred[:,1:]
    x_error = tf.exp(x_log_error)
    return tf.reduce_mean(2*x_log_error+((x_true-x_position)/x_error)**2)

h5_date = options.h5_date
h5_ext = options.h5_ext
img_ext = options.img_ext%h5_ext

# Load data
f = h5py.File('h5_files/train_x_1d_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
xpix_flat_train = f['train_x_flat'][...]
#ypix_flat_train = f['train_y_flat'][...]
cota_train = f['cota'][...]
cotb_train = f['cotb'][...]
x_train = f['x'][...] 
#y_train = f['y'][...]
clustersize_x_train = f['clustersize_x'][...]
#clustersize_y_train = f['clustersize_y'][...]

f.close()
#print(inputs_x_train[0])

perm = np.arange(len(xpix_flat_train)) 
np.random.shuffle(perm)
xpix_flat_train = xpix_flat_train[perm]
cota_train = cota_train[perm]
cotb_train = cotb_train[perm]
x_train = x_train[perm]
clustersize_x_train = clustersize_x_train[perm]
print(xpix_flat_train.shape, cota_train.shape, cotb_train.shape)
inputs_x_train = np.hstack((xpix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
#inputs_y_train = np.hstack((ypix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
angles_train = np.hstack((cota_train,cotb_train))

#h5_date = "110121"
#h5_ext = "p1_2024_irrad_BPIXL1"

f = h5py.File('h5_files/test_x_1d_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
xpix_flat_test = f['test_x_flat'][...]
#ypix_flat_test = f['test_y_flat'][...]
cota_test = f['cota'][...]
cotb_test = f['cotb'][...]
x_test = f['x'][...] 
#y_test = f['y'][...]
clustersize_x_test = f['clustersize_x'][...]
#clustersize_y_test = f['clustersize_y'][...]
inputs_x_test = np.hstack((xpix_flat_test,cota_test,cotb_test))[:,:,np.newaxis]
#inputs_y_test = np.hstack((ypix_flat_test,cota_test,cotb_test))[:,:,np.newaxis]
angles_test = np.hstack((cota_test,cotb_test))
f.close()
'''
h5_date = "082821"
h5_ext = "p1_2018_irrad_BPIXL1_file2"

# Load data
f = h5py.File('h5_files/train_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
xpix_flat_train = np.vstack((xpix_flat_train,f['train_x_flat'][...]))
ypix_flat_train = np.vstack((ypix_flat_train,f['train_y_flat'][...]))
cota_train = np.vstack((cota_train,f['cota'][...]))
cotb_train = np.vstack((cotb_train,f['cotb'][...]))
x_train = np.vstack((x_train,f['x'][...])) 
y_train = np.vstack((y_train,f['y'][...]))
clustersize_x_train = np.vstack((clustersize_x_train,f['clustersize_x'][...]))
clustersize_y_train = np.vstack((clustersize_y_train,f['clustersize_y'][...]))

inputs_x_train = np.hstack((xpix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
inputs_y_train = np.hstack((ypix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
angles_train = np.hstack((cota_train,cotb_train))
f.close()
'''
#print(angles_train.shape)
#print(xpix_flat_test[:30])
#print(ypix_flat_test[:30])
#print("clustersize of 1: ",len(np.argwhere(clustersize_x_train==1)))
for i in range(50):
#	if clustersize_x_train[i]==1:
	print(xpix_flat_train[i])
	
'''
norm_x = np.amax(xpix_flat_train)
xpix_flat_train/=norm_x
xpix_flat_test/=norm_x
norm_y = np.amax(ypix_flat_train)
ypix_flat_train/=norm_y
ypix_flat_test/=norm_y


test_c = np.zeros((13,21))
test_c[6,9]=4425
test_c[6,10]=14403
test_ang = np.array([-0.321207,-0.348136]).reshape((1,2))
print(test_ang.shape)
test_cx = test_c.sum(axis=1).reshape((1,13))
print(test_cx.shape)
test_cy = test_c.sum(axis=0).reshape((1,21))
'''
# Model configuration
batch_size = options.batch_size
loss_function = 'mse'
n_epochs_x = options.n_epochs

optimizer = Adam(lr=0.0001)
validation_split = 0.3

train_time_x = time.clock()
#train flat x

inputs = Input(shape=(13,1)) #13 in x dimension + 2 angles
angles = Input(shape=(2,))
x = Conv1D(64, kernel_size=3, padding="same")(inputs)
x = Activation("relu")(x)
x = Conv1D(64, kernel_size=3, padding="same")(x)
x = Activation("relu")(x)
x = Conv1D(64, kernel_size=2, padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling1D(pool_size=2,padding='same')(x)
x = Dropout(0.25)(x)
'''
x = Conv1D(64, kernel_size=2, padding="same")(x)
x = Activation("relu")(x)
x = Conv1D(64, kernel_size=2, padding="same")(x)
x = Activation("relu")(x) 
x = BatchNormalization(axis=-1)(x)
x = MaxPooling1D(pool_size=2,padding='same')(x)
x = Dropout(0.25)(x)
'''
x_cnn = Flatten()(x)
concat_inputs = concatenate([x_cnn,angles])
x = Dense(64)(concat_inputs)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
'''
x = Dense(128)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
'''
x = Dense(64)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x_position_logerror = Dense(2)(x)
#x_position = Activation("linear", name="x")(x)


model = Model(inputs=[inputs,angles],
              outputs=[x_position_logerror]
              )

# Display a model summary
model.summary()

#history = model.load_weights("checkpoints/cp_x%s.ckpt"%(img_ext))

# Compile the model
model.compile(loss=mse_with_errors,
              optimizer=optimizer,
              metrics=['mse']
              )

callbacks = [
EarlyStopping(patience = options.early_stop),
ModelCheckpoint(filepath="checkpoints/cp_x%s.ckpt"%(img_ext),
                save_best_only=True,
		            save_weights_only=True,
                monitor='val_loss')
]

# Fit data to model
history = model.fit([xpix_flat_train[:,:,np.newaxis],angles_train], [x_train],
                batch_size=batch_size,
                epochs=n_epochs_x,
                callbacks=callbacks,
                validation_split=validation_split)

cmsml.tensorflow.save_graph("data/graph_x_%s.pb"%(img_ext), model, variables_to_constants=True)
cmsml.tensorflow.save_graph("data/graph_x_%s.pb.txt"%(img_ext), model, variables_to_constants=True)

#plot_dnn_loss(history.history,'x',img_ext)

print("x training time for dnn",time.clock()-train_time_x)

start = time.clock()
x_pred = model.predict([xpix_flat_test[:,:,np.newaxis],angles_test], batch_size=9000)
inference_time_x = time.clock() - start
residuals_x = x_pred[:,:1] - x_test
pulls_x = residuals_x/np.exp(x_pred[:,1:])
RMS_x = np.std(residuals_x)
print("min and max residuals: ",np.amin(residuals_x),np.amax(residuals_x))
print("min and max pulls: ",np.amin(pulls_x),np.amax(pulls_x))
print("RMS_x = %f\n"%(RMS_x))


plot_residuals(residuals_x,'1dcnn','x',img_ext,type=' ')
plot_residuals(pulls_x,'1dcnn','x',img_ext,type='pulls')
#plot_by_clustersize(residuals_x,clustersize_x_test,'x',img_ext)


