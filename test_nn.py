# By: Michael Fulton
# ML Final project
# Winter 2022

from cv2 import threshold
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
    plt.xticks(np.arange(2), ["No-Sale", "Sale"], rotation=45)
    plt.yticks(np.arange(2), ["No-Sale", "Sale"])
    threshold = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color="white" if cm[i, j] > threshold else "black")
    plt.tight_layout()
    plt.ylabel('Targets')
    plt.xlabel('Predictions')
    plt.savefig("nn_conf_matrix_t-" + str(thresh) + ".png")


def calculate_accuracy(conf_matrix):
    diagonal = np.diagonal(conf_matrix)
    sum_correct_case = np.sum(diagonal)
    sum_all = np.sum(conf_matrix)
    accuracy = 0 if sum_all == 0 else (sum_correct_case / sum_all * 100)
    return accuracy

def main(train_data, train_targets, test_data, test_targets):
    # # Marketing limit determins how likely a success is for us to make the call.
    marketing_limit = 0.1

    # Create the NN model
    nn_model = Sequential([Dense(units=20, input_shape=(20,), activation='sigmoid'), Dense(
        units=20, activation='sigmoid'), Dense(units=2, activation='softmax')])

    # Compile and run the model
    nn_model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.1),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    nn_model.fit(x=train_data, y=train_targets,
                 batch_size=10, epochs=30, verbose=0)

    print('Accuracy for testing data learning_rate=0.1, momentum=0.1, and training in epochs=30')

    # Predict on the test data
    results = nn_model.predict(x=test_data, batch_size=10, verbose=0)

    # Threshold for marketing chance
    threshold_list = [0.5]

    for threshold in threshold_list:
        
        m = confusion_matrix(
            y_true=(test_targets.reshape(-1)), y_pred=(results[:,1] > (marketing_limit + threshold) ).astype(int) )
    
        print('Accuracy for threshold {0} = {1}'.format(threshold, calculate_accuracy(m)))
        
  
    print('Accuracy for testing data learning_rate=0.1, momentum=0.1, and training in epochs=30')

    # Predict on the test data
    results = nn_model.predict(x=train_data, batch_size=10, verbose=0)

    # Threshold for marketing chance

    for threshold in threshold_list:
        
        m = confusion_matrix(
            y_true=(train_targets.reshape(-1)), y_pred=(results[:,1] > (marketing_limit + threshold) ).astype(int) )
    
        print('Accuracy for threshold {0} = {1}'.format(threshold, calculate_accuracy(m)))
        


def create_data_set(draw_data):
    num_of_col = draw_data.shape[1]

    # get odd rows
    train_set = draw_data[::2, 0:num_of_col - 1]

    # get odd row of the last column
    train_target_set = draw_data[::2, num_of_col - 1:num_of_col]

    # get even rows
    test_set = draw_data[1::2, 0:num_of_col - 1]

    # get even row of last column
    test_target_set = draw_data[1::2, num_of_col - 1:num_of_col]

    return train_set, train_target_set.astype(int), test_set, test_target_set.astype(int)


def read_file(filename):
    try:
        # Read raw dataset
        raw_data = np.loadtxt(filename, delimiter=",")
        return raw_data

    # File not found
    except FileNotFoundError:
        print("Cannot read files. Please check your dataset!")
        return None


if __name__ == "__main__":
    # PATH FILES
    DATA_PATH = "./processing_dataset.csv"
    #DATA_PATH = "./processed_data.csv"

    # Read pre-processed file
    data = read_file(DATA_PATH)

    # Perform Naive Bayes algorithm
    if data is not None:
        training_set, training_target_set, testing_set, testing_target_set = create_data_set(
            data)

    main(training_set, training_target_set, testing_set, testing_target_set)
