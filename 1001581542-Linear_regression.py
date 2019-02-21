import numpy as np
import pandas as pd
import os
from numpy.linalg import inv
from matplotlib import pyplot as plt
from math import sqrt
from random import shuffle


#Name-ASHUTOSH UPADHYE
#ID-1001581542



#Calculation of Error
# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)


#Beta matrix and predicted matrix calculation
def calculate_beta_matrix(X_values, Y_values):
    b = inv(X_values.T.dot(X_values)).dot(X_values.T).dot(Y_values)
    print("Beta_Matrix:"+str(b))
    # Prediction
    yhat = X_values.dot(b)
    #print(yhat)
    return yhat, b

#R2 Score calculation
def r_square_score(actual, predicted):
    mean_y = np.mean(actual)
    ss_tot = sum((actual - mean_y) ** 2)
    ss_res = sum((actual - predicted) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def predict(X_test, b):
    y_test_predict = X_test.dot(b)
    return y_test_predict

def multiple_linear_regression():
    relative_path = os.path.abspath(os.path.dirname(__file__))
    print(relative_path)
    file_path = relative_path + "\Iris_Data.csv"
    iris_data_frame = pd.read_csv(file_path)

    length_of_records = (iris_data_frame.__len__())

 #0.5 means 50 % test data 50% training data
    num_of_rows = int(int(length_of_records) * 0.5)

  #Train-Test data split

    iris_data_frame.reindex(np.random.permutation(iris_data_frame.index))
    np.random.shuffle(iris_data_frame.values)  # shuffles data to make it random
    train_data = iris_data_frame.iloc[1:num_of_rows]  # indexes rows for training data

    test_data = iris_data_frame.iloc[num_of_rows:]  # indexes rows for test data
    train_data.sort_index()  # sorts data
    test_data.sort_index()
    # print("Test-Data")
    # print((test_data))
    # print("Train-Data")
    # print((train_data))

    #Training Data Selection
    X_values = []
    Y_values = []
    X_values_test = []
    Y_values_test = []
    for i in range(len(train_data)):
        X1 = iris_data_frame['Sepal_Length'][i]
        X2 = iris_data_frame['Sepal_Width'][i]
        X3 = iris_data_frame['Petal_Length'][i]
        X4 = iris_data_frame['Petal_Width'][i]
        Y = iris_data_frame['Species'][i]
        if Y == "Iris-setosa":
            Y = 1
        elif Y == "Iris-versicolor":
            Y = 2
        elif Y == "Iris-virginica":
            Y = 3
        features = [X1, X2, X3, X4]
        X_values.append(features.copy())
        Y_values.append(Y)
        #print(features)
        #print(Y)
    X_values = np.array(X_values, dtype='float')
    Y_values = np.array(Y_values, dtype='float')

    #print(X_values)
    #print(Y_values)
    # Beta Matrix
    predicted_matrix, beta_matrix = calculate_beta_matrix(X_values, Y_values)

    #b = inv(X_values.T.dot(X_values)).dot(X_values.T).dot(Y_values)

    ##Creating Test Data

    for j in range(len(test_data)):
        X1_test = iris_data_frame['Sepal_Length'][j]
        X2_test = iris_data_frame['Sepal_Width'][j]
        X3_test = iris_data_frame['Petal_Length'][j]
        X4_test = iris_data_frame['Petal_Width'][j]
        Y_test = iris_data_frame['Species'][j]
        if Y_test == "Iris-setosa":
            Y_test = 1
        elif Y_test == "Iris-versicolor":
            Y_test = 2
        elif Y_test == "Iris-virginica":
            Y_test = 3
        features = [X1_test, X2_test, X3_test, X4_test]
        X_values_test.append(features.copy())
        Y_values_test.append(Y_test)
        # print(features)
        # print(Y)

    X_values_test = np.array(X_values_test, dtype='float')
    Y_values_test = np.array(Y_values_test, dtype='float')

    #calculate RMSE

    rmse = rmse_metric(Y_values, predicted_matrix)

    #calculate R2_score
    r2_score = r_square_score(Y_values, predicted_matrix)

    print("Root_Mean_Square_Error:"+str(rmse))

    print("R2_score:"+str(r2_score))

    #print(X_values_test)
    predicted_test_matrix = predict(X_values_test, beta_matrix)

    #print(test_data)
    predicted_test_matrix = np.array(predicted_test_matrix, dtype=float)

    for element in range(len(predicted_test_matrix)):
        if element == 0:
            predicted_test_matrix[element] = 1

    #print(predicted_test_matrix)

    for element in range(len(predicted_test_matrix)):
        print(str(X_values_test[element])+":label-"+str(predicted_test_matrix[element]))

multiple_linear_regression()