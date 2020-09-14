#============================
# Author: Sanjana Sekhar
# Date: 13 Sep 20
#============================

import h5py
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

n_train = 30000
n_test = 3000

# Load data
f = h5py.File('train_d49301_d49341.hdf5', 'r')
input_train = f['train_hits'][...]
label_train = f['x'][...]
f.close()
f = h5py.File('test_d49350.hdf5', 'r')
input_test = f['test_hits'][...]
label_test = f['x'][...]
f.close()

#shuffle the training arrays
p = np.random.permutation(len(input_train))
input_train = input_train[p]
label_train = label_train[p]

#train/test on lesser entries for now
input_train = input_train[0:n_train]
label_train = label_train[0:n_train]

input_test = input_test[0:n_test]
label_test = label_test[0:n_test]

# Reshape data
#input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))
#input_test  = input_test.reshape((len(input_test), img_width, img_height, img_num_channels))

# Model configuration
batch_size = 32
loss_function = 'mean_squared_error'
n_epochs = 25
optimizer = Adam(lr=0.001,decay=0.0001/n_epochs)
validation_split = 0.2
verbosity = 1
        
#Conv2D -> BatchNormalization -> Pooling -> Dropout

inputs = Input(shape=(13,21,1))
x = Conv2D(16, (3, 3), padding="same")(inputs)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Dropout(0.25)(x)
x = Conv2D(32, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(32, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
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
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
model.summary()

# Compile the model
model.compile(loss=loss_function,
			  loss_weights=4.,
              optimizer=optimizer,
              metrics='mse'
              )

callbacks = [
    ModelCheckpoint("./model_checkpoint", monitor='val_loss')
]

# Fit data to model
history = model.fit(input_train, label_train,
            batch_size=batch_size,
            epochs=n_epochs,
            callbacks=callbacks,
            verbose=verbosity,
            validation_split=validation_split)

# Generate generalization metrics
x_position_predicted = model.predict(input_test)
print('R2 score for age: ', r2_score(label_test, x_position_predicted)

plt.clf()
fig = go.Figure()
fig.add_trace(go.Scattergl(
                    y=history.history['loss'],
                    name='Train'))
fig.add_trace(go.Scattergl(
                    y=history.history['val_loss'],
                    name='Valid'))
fig.update_layout(height=500, 
                  width=700,
                  title='Loss and Val loss for x position training',
                  xaxis_title='Epoch',
                  yaxis_title='Mean Absolute Error')
fig.show()
