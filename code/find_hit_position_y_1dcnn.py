#============================
# Author: Sanjana Sekhar
# Date: 17 Jan 21
#============================
'''
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
'''
import h5py
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
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
parser.add_argument("--img_ext",  default="1dcnn_%s_030524", help="file name extension for residuals and pulls")
parser.add_argument("--h5_date", default="022224", help = "date for h5 file name")
parser.add_argument("--h5_ext",  default="p1_2024_BPIX_L1U_d21601_d21800", help="h5 file name extension")
parser.add_argument("--early_stop",  default=7, type = int, help="early stopping patience (no. of epochs)")
parser.add_argument("--batch_size",  default=628, type = int, help="batch size for training")
parser.add_argument("--n_epochs",  default=30, type = int, help="no. of epochs to train for")
options = parser.parse_args()

def mse_with_errors(y_true,y_pred):
    y_position = y_pred[:,:1]
    y_log_error = y_pred[:,1:]
    y_error = tf.exp(y_log_error)
    return tf.reduce_mean(2*y_log_error+((y_true-y_position)/y_error)**2)


h5_date = options.h5_date
h5_ext = options.h5_ext
img_ext = options.img_ext%h5_ext

# Load data
f = h5py.File('h5_files/train_y_1d_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
#xpix_flat_train = f['train_x_flat'][...]
ypix_flat_train = f['train_y_flat'][...]
cota_train = f['cota'][...]
cotb_train = f['cotb'][...]
#x_train = f['x'][...] 
y_train = f['y'][...]
#clustersize_x_train = f['clustersize_x'][...]
clustersize_y_train = f['clustersize_y'][...]

f.close()
#print(inputs_x_train[0])

perm = np.arange(len(ypix_flat_train)) 
np.random.shuffle(perm)
ypix_flat_train = ypix_flat_train[perm]
cota_train = cota_train[perm]
cotb_train = cotb_train[perm]
y_train = y_train[perm]
clustersize_y_train = clustersize_y_train[perm]

inputs_y_train = np.hstack((ypix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
#inputs_y_train = np.hstack((ypix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
angles_train = np.hstack((cota_train,cotb_train))
#h5_date = "110121"
#h5_ext = "p1_2024_irrad_BPIXL1"

f = h5py.File('h5_files/test_y_1d_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
#xpix_flat_test = f['test_x_flat'][...]
ypix_flat_test = f['test_y_flat'][...]
cota_test = f['cota'][...]
cotb_test = f['cotb'][...]
#x_test = f['x'][...] 
y_test = f['y'][...]
#clustersize_x_test = f['clustersize_x'][...]
clustersize_y_test = f['clustersize_y'][...]
#inputs_x_test = np.hstack((xpix_flat_test,cota_test,cotb_test))[:,:,np.newaxis]
inputs_y_test = np.hstack((ypix_flat_test,cota_test,cotb_test))[:,:,np.newaxis]
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
#print("clustersize of 1: ",len(np.argwhere(clustersize_y_train==1)))
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
n_epochs_y = options.n_epochs
optimizer = Adamax(lr=0.001)
validation_split = 0.2


train_time_y = time.clock()

#train flat y

inputs = Input(shape=(21,1)) #13 in y dimension + 2 angles
angles = Input(shape=(2,))
y = Conv1D(32, kernel_size=3, padding="same")(inputs)
y = Activation("relu")(y)
y = Conv1D(64, kernel_size=3, padding="same")(y)
y = Activation("relu")(y)
y = MaxPooling1D(pool_size=3,padding='same')(y)
y = Dropout(0.25)(y)
y = Conv1D(64, kernel_size=3, padding="same")(y)
y = Activation("relu")(y)
y = Conv1D(32, kernel_size=3, padding="same")(y)
y = Activation("relu")(y)
#y = BatchNormalization(axis=-1)(y)
y = MaxPooling1D(pool_size=3,padding='same')(y)
y = Dropout(0.25)(y)
'''
y = Conv1D(64, kernel_size=2, padding="same")(y)
y = Activation("relu")(y)
y = Conv1D(64, kernel_size=2, padding="same")(y)
y = Activation("relu")(y) 
y = BatchNormalization(axis=-1)(y)
y = MaxPooling1D(pool_size=2,padding='same')(y)
y = Dropout(0.25)(y)
'''
y_cnn = Flatten()(y)
concat_inputs = concatenate([y_cnn,angles])
y = Dense(64)(concat_inputs)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.25)(y)

y = Dense(32)(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.25)(y)

y = Dense(32)(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.25)(y)

y_position_error = Dense(2)(y)
#y_position = Activation("linear", name="y")(y)

model = Model(inputs=[inputs,angles],
              outputs=[y_position_error]
              )

# Display a model summary
model.summary()

history = model.load_weights("checkpoints/cp_y%s.ckpt"%(img_ext))

# Compile the model
model.compile(loss=mse_with_errors,
              optimizer=optimizer,
              metrics=['mse']
              )

'''
callbacks = [
EarlyStopping(patience = options.early_stop),
ModelCheckpoint(filepath="checkpoints/cp_y%s.ckpt"%(img_ext),
                save_weights_only=True,
            		save_best_only=True,
                monitor='val_loss')
]
print(y_train.shape, y_position_error.shape)
# Fit data to model
history = model.fit([ypix_flat_train[:,:,np.newaxis],angles_train], [y_train],
                batch_size=batch_size,
                epochs=n_epochs_y,
                validation_split=validation_split,
                callbacks=callbacks)
'''
cmsml.tensorflow.save_graph("data/graph_y_%s.pb"%(img_ext), model, variables_to_constants=True)
cmsml.tensorflow.save_graph("data/graph_y_%s.pb.txt"%(img_ext), model, variables_to_constants=True)

#plot_dnn_loss(history.history,'y',img_ext)

print("y training time for dnn",time.clock()-train_time_y)

start = time.clock()
y_pred = model.predict([ypix_flat_test[:,:,np.newaxis],angles_test], batch_size=9000)
inference_time_y = time.clock() - start

#print("inference_time for dnn= ",(inference_time_x+inference_time_y))
residuals_y = y_pred[:,:1] - y_test
pulls_y = residuals_y/np.exp(y_pred[:,1:])
RMS_y = np.std(residuals_y)
print("min and max residuals: ",np.amin(residuals_y),np.amax(residuals_y))
print("min and max pulls: ",np.amin(pulls_y),np.amax(pulls_y))
print("RMS_y = %f\n"%(RMS_y))


plot_residuals(residuals_y,'1dcnn','y',img_ext,type=' ')
plot_residuals(pulls_y,'1dcnn','y',img_ext,type='pulls')

#plot_by_clustersize(residuals_y,clustersize_y_test,'y',img_ext)


