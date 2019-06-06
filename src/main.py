# Program to implement a convolutional neural network to classify hand-written
# digits from the MNIST database.

# Package imports:
import csv
from mnist import MNIST
import optparse

# Keras imports:
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense

# User imports:
import fuzzymatrix
import plot

# Main function:
## Program arguments:
parser = optparse.OptionParser()
parser.add_option('-o', action='store', dest="filename", help="filename prefix to save plot images")
options, args = parser.parse_args()

## Configuration for 'python-mnist' package:
print("Loading data... ", end='')
mndata = MNIST(path='./data', return_type='numpy')
mndata.test_img_fname = 't10k-images.idx3-ubyte'
mndata.test_lbl_fname = 't10k-labels.idx1-ubyte'
mndata.train_img_fname = 'train-images.idx3-ubyte'
mndata.train_lbl_fname = 'train-labels.idx1-ubyte'
print("(Done!)")

## Data extraction:
print("Extracting data... ", end='')

np_train_images, np_train_labels = mndata.load_training()
np_test_images, np_test_labels = mndata.load_testing()

print("(Done!)")


## Keras Model for CNN:
model = Sequential()

model.add(Conv2D(4, (5,5), padding="same", input_shape=(1,28,28)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(6, (5,5), padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dense(500))
model.add(Dense(10))
model.add(Activation("softmax"))
