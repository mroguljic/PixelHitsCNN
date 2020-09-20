#============================
# Author: Sanjana Sekhar
# Date: 13 Sep 20
#============================

import h5py
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Concatenate
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

#n_train = 41000
#n_test = 4100

# Load data
f = h5py.File('h5_files/train_subset.hdf5', 'r')
pix_train = f['train_hits'][...]
cosx_train = f['cosx'][...]
cosy_train = f['cosy'][...]
cosz_train = f['cosz'][...]
label_train = f['x'][...]
f.close()
f = h5py.File('h5_files/test_subset.hdf5', 'r')
pix_test = f['test_hits'][...]
cosx_test = f['cosx'][...]
cosy_test = f['cosy'][...]
cosz_test = f['cosz'][...]
label_test = f['x'][...]
f.close()

#print(np.reshape(pix_train[2],(13,21)))
print(np.reshape(cosx_train,(41000)))
'''
pix_train = np.zeros((n_train,13,21,1))
label_train = np.zeros((n_train,1))
#shuffle the training arrays -> FIND MORE EFFICIENT WAY TO DO THIS
#create subset
j=0
for i in range(0,41):
	pix_train[1000*i:1000*(i+1)]=pix_train_1[j:1000+j]
	label_train[1000*i:1000*(i+1)]=label_train_1[j:1000+j]
	j+=30000

pix_test = pix_test_1[0:n_test]
label_test = label_test_1[0:n_test]
'''
# Model configuration
batch_size = 64
loss_function = 'mean_squared_error'
n_epochs = 1
optimizer = Adam(lr=0.001)
validation_split = 0.2
verbosity = 1
        
#Conv2D -> BatchNormalization -> Pooling -> Dropout

inputs = Input(shape=(13,21,1))
x = Conv2D(32, (3, 3), padding="same")(inputs)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
angles = Input(shape=(3))
x = Dense(128)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1)(x)
x_position = Activation("linear", name="x_position")(x)

model = Model(inputs=inputs,
              outputs=x_position
              )

# Display a model summary
model.summary()

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['mse']
              )

callbacks = [
    ModelCheckpoint(filepath="checkpoints/cp.ckpt", monitor='val_loss')
]

# Fit data to model
history = model.fit(pix_train, label_train,
            batch_size=batch_size,
            epochs=n_epochs,
            callbacks=callbacks,
            verbose=verbosity,
            validation_split=validation_split)

''''
# Generate generalization metrics
results = model.predict(pix_test, batch_size=batch_size)
#print("test loss, test acc:", results)
print results[:20]
residuals = results-label_test

pylab.plot(history.history['loss'])
pylab.plot(history.history['val_loss'])
pylab.title('model loss')
pylab.ylabel('loss')
pylab.xlabel('epoch')
pylab.legend(['train', 'validation'], loc='upper right')
#pylab.show()
pylab.savefig("pixelcnn_x.png")
pylab.close()

plt.hist(residuals, histtype='step')
plt.title(r'$\vartriangle x = x_{pred} - x_{true}$')
plt.ylabel('No. of samples')
plt.xlabel(r'$\vartriangle x$')
plt.show()
plt.savefig("plots/x_residuals_sep19.png")

'''