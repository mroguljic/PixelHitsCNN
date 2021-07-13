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


h5_date = "dec12"
h5_ext = "phase1"
img_ext = "2dcnn_p1_jul13_test"

# Load data
f = h5py.File('h5_files/train_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
pix_train = (f['train_hits'][...])/35000
cota_train = f['cota'][...]
cotb_train = f['cotb'][...]
x_train = f['x'][...] 
y_train = f['y'][...]
clustersize_x_train = f['clustersize_x'][...]
clustersize_y_train = f['clustersize_y'][...]
angles_train = np.hstack((cota_train,cotb_train))
f.close()

f = h5py.File('h5_files/test_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
pix_test = (f['test_hits'][...])/35000
cota_test = f['cota'][...]
cotb_test = f['cotb'][...]
x_test = f['x'][...]
y_test = f['y'][...]
clustersize_x_test = f['clustersize_x'][...]
clustersize_y_test = f['clustersize_y'][...]
angles_test = np.hstack((cota_test,cotb_test))
f.close()


print("max train = ",np.amax(pix_train))
print("max test = ",np.amax(pix_test))
#pix_train/=np.amax(pix_train)
#pix_test/=np.amax(pix_train) 
   


# Model configuration
batch_size = 256
loss_function = 'mse'
n_epochs = 5
optimizer = Adam(lr=0.001)
validation_split = 0.2

train_time_s = time.clock()
#Conv2D -> BatchNormalization -> Pooling -> Dropout



inputs = Input(shape=(13,21,1))
#angles = Input(shape=(2,))
x = Conv2D(16, (3, 3), padding="same")(inputs)
x = Activation("relu")(x)
#x = BatchNormalization(axis=-1)(x)

x_cnn = Flatten()(x)
#concat_inputs = concatenate([x_cnn,angles])

#x = Dense(32)(concat_inputs)
#x = Activation("relu")(x)
x = Dense(1)(x_cnn)
x_position = Activation("linear", name="x")(x)

#y = Dense(32)(concat_inputs)
#y = Activation("relu")(y)
#y = Dense(1)(y)
#y_position = Activation("linear", name="y")(y)

model = Model(inputs=[inputs],#angles],
              outputs=[x_position]#,y_position]
              )

 # Display a model summary
model.summary()

#history = model.load_weights("checkpoints/cp_%s.ckpt"%(img_ext))

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['mse']#,'mse']
              )



callbacks = [
EarlyStopping(patience=5),
ModelCheckpoint(filepath="checkpoints/cp_%s.ckpt"%(img_ext),
                save_weights_only=True,
                monitor='val_loss')
]

# Fit data to model
#dividing all inputs by 35k to keep it in a range
history = model.fit([pix_train], #angles_train],
		[x_train],# y_train],
                batch_size=batch_size,
                epochs=n_epochs,
                callbacks=callbacks,
                validation_split=validation_split)

cmsml.tensorflow.save_graph("data/graph_%s.pb"%(img_ext), model, variables_to_constants=True)
cmsml.tensorflow.save_graph("data/graph_%s.pb.txt"%(img_ext), model, variables_to_constants=True)

# Generate generalization metrics
print("training time ",time.clock()-train_time_s)

start = time.clock()
x_pred = model.predict([pix_test], batch_size=9000)
inference_time = time.clock() - start

print("inference_time = ",inference_time)

'''

model = tf.saved_model.load("data/")

start = time.clock()
x_pred_saved, y_pred_saved = model.predict([pix_test, angles_test], batch_size=9000)
inference_time = time.clock() - start

print("inference_time = ",inference_time)
'''
for cl in range(10):

   print((pix_test[cl]).reshape((13,21)))
   print('x_label = %f, y_label = %f, cota = %f, cotb = %f\n'%(x_test[cl],y_test[cl], cota_test[cl],cotb_test[cl]))
   print('x_pred = %f, y_pred = \n'%(x_pred[cl]))#,y_pred[cl]))
   #print('x_pred_saved = %f, y_pred_saved = %f\n'%(x_pred_saved[cl],y_pred_saved[cl]))
   print("\n")

print("====================================================================================")
