# Program to implement a convolutional neural network to classify hand-written
# digits from the MNIST database.

# Package imports:
import optparse

# Keras imports:
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten
from keras import backend as K

# User imports:
import fuzzymatrix
import plot

# Main function:

## Program arguments:
parser = optparse.OptionParser()
parser.add_option('-o', action='store', dest="filename", help="filename prefix to save plot images")
options, args = parser.parse_args()

## Training settings:
batch_size = 128
img_rows, img_cols = 28, 28
num_classes = 10
epochs = 2

## Data extraction:
print("Extracting data...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Data extracted!")

## Data formatting:
print("Formatting data...")

### Format backend data format:
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

### Format inputs:
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train /= 255
x_test /= 255

### Format outputs:
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("Data formatted!")

## Keras Model for CNN:
model = Sequential()

model.add(Conv2D(4, (5,5), padding="same", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(6, (5,5), padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

## Train model:
print("Training model...")

training_history = model.fit(x_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=1,
                             validation_data=(x_test, y_test))

print("Model trained!")

## Evaluate results:
score = model.evaluate(x_test, y_test, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
