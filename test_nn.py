# By: Tri Le
# ML Final project
# Winter 2022
# Reference: nn.py by Michael Fulton

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix



def calculate_accuracy(conf_matrix):
    diagonal = np.diagonal(conf_matrix)
    sum_correct_case = np.sum(diagonal)
    sum_all = np.sum(conf_matrix)
    accuracy = 0 if sum_all == 0 else (sum_correct_case / sum_all * 100)

    return accuracy


def run_test(train_data, train_targets, test_data, test_targets):
    # # Marketing limit determins how likely a success is for us to make the call.
    marketing_limit = 0.1
    threshold = 0.5
    # Create the NN model
    nn_model = Sequential([Dense(units=20, input_shape=(20,), activation='sigmoid'), Dense(
        units=20, activation='sigmoid'), Dense(units=2, activation='softmax')])

    # Compile and run the model
    nn_model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.1),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    nn_model.fit(x=train_data, y=train_targets,
                 batch_size=10, epochs=30, verbose=0)

    print('Accuracy for training data learning_rate=0.1, momentum=0.1, and training in epochs=30')

    # Predict on the test data
    results = nn_model.predict(x=train_data, batch_size=10, verbose=0)

    cf_matrix = confusion_matrix(
        y_true=(train_targets.reshape(-1)), y_pred=(results[:, 1] > (marketing_limit + threshold)).astype(int))

    print('Accuracy training data = {0}'.format(calculate_accuracy(cf_matrix)))

    print('Accuracy for testing data learning_rate=0.1, momentum=0.1, and training in epochs=30')

    # Predict on the test data
    results = nn_model.predict(x=test_data, batch_size=10, verbose=0)

    # Threshold for marketing chance
    cf_matrix = confusion_matrix(
        y_true=(test_targets.reshape(-1)), y_pred=(results[:, 1] > (marketing_limit + threshold)).astype(int))

    print('Accuracy training data = {0}'.format(calculate_accuracy(cf_matrix)))


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

    # Read pre-processed file
    data = read_file(DATA_PATH)

    # Perform Naive Bayes algorithm
    if data is not None:
        training_set, training_target_set, testing_set, testing_target_set = create_data_set(
            data)

    run_test(training_set, training_target_set,
             testing_set, testing_target_set)
