import h5py
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.losses import mean_squared_error as mse
from keras.optimizers import Adam

# Model configuration
batch_size = 1000
img_width, img_height, img_num_channels = 13, 21, 1
loss_function = 'mean_squared_error'
#no_classes = 10
no_epochs = 25
optimizer = Adam()
validation_split = 0.2
verbosity = 1

# Load MNIST data
f = h5py.File('train_d49301.hdf5', 'r')
input_train = f['train_hits'][...]
label_train = f['x'][...]
f.close()
f = h5py.File('test_d49350_smol.hdf5', 'r')
input_test = f['test_hits'][...]
label_test = f['x'][...]
f.close()

# Reshape data
#input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))
#input_test  = input_test.reshape((len(input_test), img_width, img_height, img_num_channels))

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))

# Display a model summary
model.summary()

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer
              )

# Fit data to model
history = model.fit(input_train, label_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(input_test, label_test, verbose=0)
print('Test: %f'%(score))

