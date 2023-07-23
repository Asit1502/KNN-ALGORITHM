# KNN-ALGORITHM
1.Implementation of K-NN classification algorithm with Euclidean and Manhattan Distance metrics,that work for any k values on Iris dataset.
2.Modified K-NN implementation for regression problem. 

This code demonstrates the implementation and comparison of K-Nearest Neighbors (K-NN) classification and regression algorithms using custom implementations and scikit-learn library. The dataset used in the code is the Iris dataset for classification and a custom-generated dataset for regression.

Here's a brief algorithm for each part of the code:

Part 1: K-Nearest Neighbors (K-NN) Classification

1.Import necessary libraries: Import numpy, sklearn.datasets, sklearn.model_selection, and sklearn.metrics.

2.Define Euclidean and Manhattan distance functions: The euclidean_distance function calculates the Euclidean distance between two points, and the manhattan_distance function calculates the Manhattan distance between two points.

3.Define the K-NN classification algorithm: The k_nearest_neighbors function takes the training data (X_train and y_train), testing data (X_test), the number of nearest neighbors (k), and the distance metric (either "euclidean" or "manhattan") as inputs. It iterates through each test data point, calculates the distances between the test point and all training points, selects the k nearest neighbors based on the chosen distance metric, and predicts the label for the test point by choosing the majority class among its nearest neighbors.

4.Load the Iris dataset and split it into training and testing sets.

5.Set the value of k and the distance metric.

6.Apply the custom K-NN algorithm (k_nearest_neighbors) to the data, make predictions on the test data, and calculate the accuracy of the predictions using scikit-learn's accuracy_score function.

PART 2: K-Nearest Neighbors (K-NN) Regression

1.Import necessary libraries: Import numpy, pandas, sklearn.model_selection, sklearn.neighbors, and sklearn.metrics.

2.Load the custom Iris dataset from a CSV file using pandas.

3.Extract the sepal length (X) and sepal width (y) features from the dataset.

4.Split the data into training and testing sets.

5.Define the K-NN regression algorithm: The k_nearest_neighbors_regression function takes the training data (X_train and y_train), testing data (X_test), and the number of nearest neighbors (k) as inputs. For each test data point, it calculates the distances to all training points, selects the k nearest neighbors, and performs a weighted average of the y values of those neighbors based on the inverse of their distances. This weighted average is then used as the prediction for the test data point.

6.Set the values of k for K-NN regression (k_values).

7.Apply the custom K-NN regression algorithm (k_nearest_neighbors_regression) for each value of k and calculate the Root Mean Squared Error (RMSE) using scikit-learn's mean_squared_error function.

8.Apply scikit-learn's K-NN regression (KNeighborsRegressor) for each value of k, fit the model to the training data, make predictions on the test data, and calculate the RMSE.

Overall, the code provides a custom implementation of the K-NN classification and regression algorithms and compares them with scikit-learn's implementation using Iris dataset for classification and a custom-generated dataset for regression.
