from pickletools import uint8
import numpy as np
import os
import matplotlib.pyplot as plt


rng = np.random.default_rng()

data_folder = '0.001'

N = 10  # number of perceptron
eta = 0.001

# prod_weigh_file = './weights.txt'
# prod_training_file = './mnist_train.csv'
# test_training_file = './train_test.csv'
# prod_test_file = './mnist_test.csv'


def training_and_test(epoch, training_data_file, test_data_file, training_result_file, testing_result_file):
    current_epoch = 0
    weights = None
    training_data_matrix = None
    test_data_matrix = None

    training_result = np.array([0]).astype(float)
    test_result = np.array([0]).astype(float)
    epoch_array = np.array([0]).astype(float)
    while current_epoch < epoch:
        current_epoch += 1

        weights, training_data_matrix = training_data(
            training_data_file, './{0}/result_weights.txt'.format(data_folder), weights=weights, previous_data=training_data_matrix)

        traning_data_test_result, training_data_matrix = test_data('./{0}/result_weights.txt'.format(data_folder),
                                                                   training_data_file, './{0}/training_test_result.txt'.format(data_folder), weights=weights, previous_data=training_data_matrix)
        training_result = np.append(training_result, traning_data_test_result)

        test_data_test_result, test_data_matrix = test_data('./{0}/result_weights.txt'.format(data_folder),
                                                            test_data_file, './{0}/test_test_result.txt'.format(data_folder), weights=weights, previous_data=test_data_matrix)
        test_result = np.append(test_result, test_data_test_result)

        epoch_array = np.append(epoch_array, float(current_epoch))

        np.savetxt(training_result_file, np.array([training_result, epoch_array]),  delimiter=', ',
                   newline='\n', header='', footer='', comments='# ', encoding=None)

        np.savetxt(testing_result_file, np.array([test_result, epoch_array]), delimiter=', ',
                   newline='\n', header='', footer='', comments='# ', encoding=None)

        print('epoch = {0} , training accuracy = {1}, test accuracy = {2}'.format(
            current_epoch, traning_data_test_result, test_data_test_result))

    return [training_result, test_result]


def draw_chart(training_result_file, test_result_file, figure_name):
    test_result = np.loadtxt(test_result_file,  delimiter=', ')
    training_result = np.loadtxt(training_result_file,  delimiter=', ')
    # i = 0
    # while  i<  10 or i <len(test_result[0]):
    #     if  test_result[:,i]< 0.5:
    #         test_result = np.delete(test_result, i, axis=1)
    #         training_result = np.delete(training_result, i, axis=1)
    #     else:
    #         i = i + 1

    # ax=plt.gca()

    # f = lambda x,pos: str(x).rstrip('0').rstrip('.')
    # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))

    plt.plot(test_result[1, :], test_result[0, :],
             label="Accuracy on the test data" )
    plt.plot(training_result[1, :], training_result[0, :],
             label="Accuracy on the training data")
    plt.grid()
    plt.legend()
    plt.ylabel('accuracy(%)')
    plt.xlabel('epoch')
    plt.title("The result with \u03B7 = {0}".format(eta))
    # plt.show()
    plt.savefig(figure_name)


def training_data(training_file, weight_file, weights=None, previous_data=None):
    target = None
    input_data = None

    if previous_data is None:
        print('Load data from file')
        draw_data = np.genfromtxt(
            training_file, delimiter=',', skip_header=True)
        target = draw_data[:, 0].astype(int)
        input_data = draw_data[:, 1:].astype(np.float64)
        input_data = input_data / 255
        
        print(input_data[input_data > 1])

        data_for_bias = np.full((input_data.shape[0]), 1.0).astype(np.float64)
        input_data = np.insert(input_data, 0, data_for_bias, axis=1)
    else:
        target, input_data = previous_data

    num_of_column = input_data.shape[1]
    num_of_row = input_data.shape[0]

    print('...training: the number of input = {0}'.format(num_of_row))

    if weights is None:
        print('Generate weights')

        if os.path.exists(weight_file) == False:
            weights = rng.random((N, num_of_column), dtype=np.float64)
            weights = weights - 0.5
        else:
            weights = np.loadtxt(weight_file,  delimiter=', ')

    for i in range(0, num_of_row):

        for j in range(0, N):
            activation = np.dot(input_data[i, :], weights[j, :])
            y = 1 if activation > 0 else 0
            binary_target = 1 if j == target[i] else 0

            if y-binary_target != 0:
                weights[j, :] += eta * (binary_target - y)*input_data[i, :]

    np.savetxt(weight_file, weights, delimiter=', ', newline='\n',
               header='', footer='', comments='# ', encoding=None)

    return [weights, [target, input_data]]


def test_data(weight_file, test_file, test_result, weights=None, previous_data=None):
    target = None
    input_data = None

    if weights is None:
        print('Load weights from file')
        weights = np.loadtxt(weight_file,  delimiter=',')

    if previous_data is None:
        print('Load test data from file')
        draw_data = np.genfromtxt(
            test_file, delimiter=',', skip_header=True)
        target = draw_data[:, 0].astype(int)

        input_data = draw_data[:, 1:].astype(np.float64)
        input_data = input_data / 255

        data_for_bias = np.full(
            (input_data.shape[0]), 1.0).astype(np.float64)
        input_data = np.insert(input_data, 0, data_for_bias, axis=1)
    else:
        target, input_data = previous_data

    num_of_row = input_data.shape[0]
    print('...testing: the number of input = {0}'.format(num_of_row))

    confusion_matrix = np.full((10, 10), 0).astype(int)

    output = 0
    for i in range(0, num_of_row):

        for j in range(0, N):
            output = 1 if np.dot(input_data[i, :], weights[j, :]) > 0 else 0
            confusion_matrix[target[i], j] += output

    diagonal = np.diagonal(confusion_matrix)
    sum_correct_case = np.sum(diagonal)
    sum_all = np.sum(confusion_matrix)

    np.savetxt(test_result, confusion_matrix, fmt='%i', delimiter='\t',
               newline='\n', header='', footer='', comments='# ', encoding=None)

    return [sum_correct_case/sum_all * 100, [target, input_data]]


training_and_test(70,  prod_training_file, prod_test_file,
                  './{0}/training_result_statistic.txt'.format(data_folder), './{0}/test_result_statistic.txt'.format(data_folder))

# training_and_test(10, test_training_file, test_training_file,
#                   './{0}/training_result_statistic.txt'.format(data_folder), './{0}/test_result_statistic.txt'.format(data_folder))

draw_chart('./{0}/training_result_statistic.txt'.format(data_folder),
           './{0}/test_result_statistic.txt'.format(data_folder), './{0}/chart.png'.format(data_folder))

# weight = training_data(prod_training_file, 'weights.txt')
# test_data(prod_weigh_file, prod_test_file, './result.txt', weight)
