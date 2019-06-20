# Program to implement a convolutional neural network to classify hand-written
# digits from the MNIST database.

# Package imports:
import argparse

# Keras imports:
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten

# User imports:
import batchlog
import csvconverter
import plot

# Main function:

## Program arguments:
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', dest='batch_size', type=int,
                    default=128,
                    help="batch size used for training")
parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=10,
                    help="number of training epochs")
parser.add_argument('-bf', '--batch-file', dest='filename_batch',
                    default='results_batch.csv',
                    help="filename of batch history csv output")
parser.add_argument('-tf', '--train-file', dest='filename_train',
                    default='results_train.csv',
                    help="filename of training history csv output")
parser.add_argument('-wf', '--weights-file', dest='filename_weights',
                    help="load CNN weights from a file instead of training")
args = parser.parse_args()

## Training settings:
img_rows, img_cols = 28, 28
num_classes = 10

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

model.add(Conv2D(4, (5,5), padding="same", input_shape=input_shape, name="f1"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(6, (5,5), padding="same", name="f2"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

## Print model information:
print("Model summary:")
print(model.summary())

if(args.filename_weights):

    print("Loading weights...")
    model.load_weights(args.filename_weights)
    print("Weights loaded!")

else:

    ## Train model:
    print("Training model...")

    batch_history = batchlog.BatchLog()

    training_history = model.fit(x_train, y_train,
                                 batch_size=args.batch_size,
                                 epochs=args.epochs,
                                 verbose=1,
                                 validation_data=(x_test, y_test),
                                 callbacks=[batch_history])

    print("Model trained!")

    ## Save training history in .csv file
    csvconverter.savecsv(csvconverter.converter(training_history.history),
                         args.filename_train)

    ## Save batch history in .csv file
    csvconverter.savecsv(batch_history.log, args.filename_batch)

## Evaluate results:
score = model.evaluate(x_test, y_test, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])


