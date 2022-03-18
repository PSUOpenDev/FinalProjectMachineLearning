# By: Tri Le
# ML Final project
# Winter 2022

from naive_bayes import read_file, create_data_set, train_and_test

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

        train_and_test(training_set, training_target_set,
                       testing_set, testing_target_set)
