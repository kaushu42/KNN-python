import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from knn import KNN

# Load the dataset
X = pd.read_csv('./data.csv')

# Shuffle the data
X = shuffle(X)

# Get the target value from the data
y = X['target']
# We no longer need a target column in the dataframe
del X['target']

# Convert dataframes to numpy arrays
X = X.values
y = y.values

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y)

# We no longer need X and y
del X, y

# Create a KNN classifier
classifier = KNN(k=5)

# Train the classifier
classifier.train(x_train, y_train)

# This will store the predictions
predictions = []

# Only show 10 plots
show_n_examples = 10

# Loop through the test set to predict class for each point
for i in range(len(y_test)):
    # Get predictions
    pred = classifier.predict(x_test[i])
    predictions.append(pred)

    if show_n_examples > 0:
        print('Class =', pred)
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
        plt.scatter(x_test[i, 0], x_test[i, 1], c=['r'])
        plt.show()
        show_n_examples -= 1

predictions = np.asarray(predictions)

# Calculate accuracy of the classifier
print('Accuracy: {0}%'.format(np.mean(predictions== y_test)*100))
# print(y_test[a])
# a[a[:, 0].argsort()]
