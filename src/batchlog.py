# Module to handle batch operations.

# Package imports:
import keras
from keras.callbacks import Callback

## Callbacks to be executed on batch and and on train began
class BatchLog(keras.callbacks.Callback):

  ## Declares log list
  def on_train_begin(self, logs={}):
    self.log = []

  ## Append batch log information
  def on_batch_end(self, batch, logs={}):
    self.log.append(logs)
