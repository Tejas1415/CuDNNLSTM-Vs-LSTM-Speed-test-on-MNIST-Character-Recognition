import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM ## CuDNNLSTM is superfast and trains with very very high accuracy

#################### Pre-Processing #################################
mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

x_train = x_train/255.0
x_test = x_test/255.0

print(x_train.shape)
print(x_train[0].shape)

############################# Building The Model #####################
model = Sequential()

### Uncomment the lines to see the results for an LSTM layer. Currently executing CuDNNLSTM network.

# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
#model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(CuDNNLSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True)) ### cudacnn uses tanh activation function automatically

model.add(Dropout(0.2))

#model.add(LSTM(128, activation='relu'))
model.add(CuDNNLSTM(128))

model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

### Declare an Optimizer - Adam Optimizer works well
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

########## CuDNNLSTM's could get 99% in 1 min 15 secs. Whereas normal LSTM networks took approx 8 mins to achieve 94% accuracy #######
model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test))
