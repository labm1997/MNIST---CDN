# Program to implement a convolutional neural network to classify hand-written
# digits from the MNIST database.

# Package imports:
import csv
from mnist import MNIST
import optparse

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
