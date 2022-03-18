from kmean import accuracy,k_means_train,k_means_test,k_def,r_def,preprocess
from naive_bayes import create_data_set,read_file




if __name__ == "__main__":
    raw_data = read_data("./bank-additional-full.csv")
    preprocess(raw_data)
    # split data for testing and training
    test_set = raw_data.sample(raw_data.shape[0] % 10000)
    test_data = test_set.drop(columns='y')
    train_set = raw_data.drop(test_set.index)
    train_data = train_set.drop(columns='y')


        # PATH FILES
    DATA_PATH = "./processing_dataset.csv"


    # Read pre-processed file
    data = read_file(DATA_PATH)

    # Perform Naive Bayes algorithm
    if data is not None:
        training_set, training_target_set, testing_set, testing_target_set = create_data_set(
            data)

        # training
        errors, centroids, clusters, train_predict = k_means_train(training_set, k_def, r_def)
        train_accuracy, case = accuracy(training_target_set.tolist(), train_predict)
        print("Train Accuracy:", train_accuracy)
        # testing
        test_predict = k_means_test(centroids, testing_set, k_def)
        test_accuracy = accuracy(testing_target_set.tolist(), test_predict, case)
        print("Test Accuracy:", test_accuracy)
        print("Confusion Matrix:")
