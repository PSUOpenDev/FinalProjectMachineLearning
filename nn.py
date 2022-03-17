# By: Michael Fulton
# ML Final project
# Winter 2022

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pylab as plt


def save_confusion_matrix(cm, thresh):
  plt.clf()
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Bank Marketing Prediction Results")
  plt.colorbar()
  plt.xticks(np.arange(2), ["No-Sale", "Sale"], rotation = 45)
  plt.yticks(np.arange(2), ["No-Sale", "Sale"])
  threshold = cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i,cm[i,j], horizontalalignment='center', color="white" if cm[i,j] > threshold else "black")
  plt.tight_layout()
  plt.ylabel('Targets')
  plt.xlabel('Predictions')
  plt.savefig("nn_conf_matrix_t-" + str(thresh) + ".png")

def main():
  #Marketing limit determins how likely a success is for us to make the call.
  marketing_limit = 0.1

  #Load normalized data
  pos = np.load("normalized_pos_data.npy")
  neg = np.load("normalized_neg_data.npy")

  #Shuffle Data
  np.random.shuffle(pos)
  np.random.shuffle(neg)

  #Split into training and testing sets. 50 50
  pos_train, pos_test = np.split(pos, [len(pos) // 2], axis=0)
  neg_train, neg_test = np.split(neg, [len(neg) // 2], axis=0)


  #Put neg and pos together, shuffle data
  train_w_targets = np.concatenate((pos_train, neg_train), axis=0)
  test_w_targets = np.concatenate((pos_test, neg_test), axis=0)
  np.random.shuffle(train_w_targets)
  np.random.shuffle(test_w_targets)
 
  #Split targets out of data
  train_data, train_targets = np.split(train_w_targets, [-1], axis=1)
  test_data, test_targets = np.split(test_w_targets, [-1], axis=1)

  #Convert targets to integers, so it can more easily be parsed for the confusion matrix
  test_targets = test_targets.astype(int)

  #Create the NN model
  nn_model = Sequential([Dense(units=64, input_shape=(64,), activation='sigmoid'), Dense(units=64, activation='sigmoid'), Dense(units=2, activation='softmax')])

  #Compile and run the model
  nn_model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  nn_model.fit(x=train_data, y=train_targets, batch_size=10, epochs=30, verbose=0)

  #Predict on the test data
  results = nn_model.predict(x=test_data, batch_size=10, verbose=0)
  
  
  #Threshold for marketing chance
  thresh_results = np.arange(len(results), dtype=int)
  for i in range(len(results)):
    if results[i][1] > marketing_limit + 0.3:
      thresh_results[i] = 1
    else:
      thresh_results[i] = 0

  thresh_results2 = np.arange(len(results), dtype=int)
  for i in range(len(results)):
    if results[i][1] > marketing_limit + 0.2:
      thresh_results2[i] = 1
    else:
      thresh_results2[i] = 0

  thresh_results3 = np.arange(len(results), dtype=int)
  for i in range(len(results)):
    if results[i][1] > marketing_limit + 0.1:
      thresh_results3[i] = 1
    else:
      thresh_results3[i] = 0

  thresh_results4 = np.arange(len(results), dtype=int)
  for i in range(len(results)):
    if results[i][1] > marketing_limit:
      thresh_results4[i] = 1
    else:
      thresh_results4[i] = 0

  #Create the conf_matrix for thresholding on 
  conf_matrix = confusion_matrix(y_true=(test_targets.reshape(-1)), y_pred=(np.argmax(results, axis=-1)))
  conf_matrix2 = confusion_matrix(y_true=(test_targets.reshape(-1)), y_pred=thresh_results)
  conf_matrix3 = confusion_matrix(y_true=(test_targets.reshape(-1)), y_pred=thresh_results2)
  conf_matrix4 = confusion_matrix(y_true=(test_targets.reshape(-1)), y_pred=thresh_results3)
  conf_matrix5 = confusion_matrix(y_true=(test_targets.reshape(-1)), y_pred=thresh_results4)

  #Save graphical version of confusion matrix
  save_confusion_matrix(conf_matrix, 0.5)
  save_confusion_matrix(conf_matrix2, 0.4)
  save_confusion_matrix(conf_matrix3, 0.3)
  save_confusion_matrix(conf_matrix4, 0.2)
  save_confusion_matrix(conf_matrix5, 0.1)

if __name__ == "__main__":
  main()