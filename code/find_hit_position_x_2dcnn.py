#============================
# Author: Sanjana Sekhar
# Date: 13 Sep 20
#============================

'''
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
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
import cmsml

'''
h5_date = "082821"
h5_ext = "p1_2018_irrad_BPIXL1"
img_ext = "2dcnn_%s_aug31"%h5_ext
'''
h5_date = "110121"
h5_ext = "p1_2024_irrad_BPIXL1"
img_ext = "2dcnn_%s_nov11"%h5_ext

# Load data
f = h5py.File('h5_files/train_x_2d_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
pix_train = (f['train_hits'][...])
cota_train = f['cota'][...]
cotb_train = f['cotb'][...]
x_train = f['x'][...] 
#y_train = f['y'][...]
clustersize_x_train = f['clustersize_x'][...]
#clustersize_y_train = f['clustersize_y'][...]
angles_train = np.hstack((cota_train,cotb_train))
f.close()

perm = np.arange(len(pix_train)) 
np.random.shuffle(perm)
pix_train = pix_train[perm]
cota_train = cota_train[perm]
cotb_train = cotb_train[perm]
x_train = x_train[perm]
clustersize_x_train = clustersize_x_train[perm]
print(pix_train.shape, cota_train.shape, cotb_train.shape)
#inputs_x_train = np.hstack((xpix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
#inputs_y_train = np.hstack((ypix_flat_train,cota_train,cotb_train))[:,:,np.newaxis]
angles_train = np.hstack((cota_train,cotb_train))

f = h5py.File('h5_files/test_x_2d_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
pix_test = (f['test_hits'][...])
cota_test = f['cota'][...]
cotb_test = f['cotb'][...]
x_test = f['x'][...]
#y_test = f['y'][...]
clustersize_x_test = f['clustersize_x'][...]
#clustersize_y_test = f['clustersize_y'][...]
angles_test = np.hstack((cota_test,cotb_test))
f.close()
'''
h5_date = "082821"
h5_ext = "p1_2018_irrad_BPIXL1_file2"

# Load data
f = h5py.File('h5_files/train_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
pix_train = np.vstack((pix_train,f['train_hits'][:3000000]))
cota_train = np.vstack((cota_train,f['cota'][:3000000]))
cotb_train = np.vstack((cotb_train,f['cotb'][:3000000]))
x_train = np.vstack((x_train,f['x'][:3000000])) 
y_train = np.vstack((y_train,f['y'][:3000000]))
clustersize_x_train = np.vstack((clustersize_x_train,f['clustersize_x'][:3000000]))
clustersize_y_train = np.vstack((clustersize_y_train,f['clustersize_y'][:3000000]))

angles_train = np.hstack((cota_train,cotb_train))
'''
#print("max train = ",np.amax(pix_train))
#print("max test = ",np.amax(pix_test))
#pix_train/=np.amax(pix_train)
#pix_test/=np.amax(pix_train) 
   


# Model configuration
batch_size = 512
loss_function = 'mse'
n_epochs_x = 20
n_epochs_y = 20
optimizer = Adam(lr=0.001)
validation_split = 0.2

train_time_x = time.clock()
#Conv2D -> BatchNormalization -> Pooling -> Dropout



inputs = Input(shape=(13,21,1))
angles = Input(shape=(2,))
x = Conv2D(64, (3, 3), padding="same")(inputs)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)
x = Dropout(0.25)(x)
'''
#x = Conv2D(256, (3, 3), padding="same")(x)
#x = Activation("relu")(x)
#x = BatchNormalization(axis=-1)(x)
#x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)
#x = Dropout(0.25)(x)
'''
x = Conv2D(64, (2, 2), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)
x = Dropout(0.25)(x)

x_cnn = Flatten()(x)
concat_inputs = concatenate([x_cnn,angles])

x = Dense(64)(concat_inputs)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(128)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
'''
#x = Dense(256)(x)
#x = Activation("relu")(x)
#x = BatchNormalization()(x)
#x = Dropout(0.25)(x)

x = Dense(128)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
'''
x = Dense(64)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(1)(x)
x_position = Activation("linear", name="x")(x)


model = Model(inputs=[inputs,angles],
              outputs=[x_position]
              )

# Display a model summary
model.summary()

history = model.load_weights("checkpoints/cp_x%s.ckpt"%(img_ext))

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['mse']
              )
'''
callbacks = [
EarlyStopping(patience=7),
ModelCheckpoint(filepath="checkpoints/cp_x%s.ckpt"%(img_ext),
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss')
]

# Fit data to model
history = model.fit([pix_train, angles_train], [x_train],
                batch_size=batch_size,
                epochs=n_epochs_x,
                callbacks=callbacks,
                validation_split=validation_split)
'''
cmsml.tensorflow.save_graph("data/graph_x_%s.pb"%(img_ext), model, variables_to_constants=True)
cmsml.tensorflow.save_graph("data/graph_x_%s.pb.txt"%(img_ext), model, variables_to_constants=True)

#plot_dnn_loss(history.history,'x',img_ext)

print("x training time for 2dcnn",time.clock()-train_time_x)

start = time.clock()
x_pred = model.predict([pix_test, angles_test], batch_size=9000)
inference_time_x = time.clock() - start

residuals_x = x_pred - x_test
RMS_x = np.std(residuals_x)
print(np.amin(residuals_x),np.amax(residuals_x))
print("RMS_x = %f\n"%(RMS_x))

plot_residuals(residuals_x,'2dcnn','x',img_ext)
#plot_by_clustersize(residuals_x,clustersize_x_test,'x',img_ext)
