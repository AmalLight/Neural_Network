from keras.models      import Sequential
from keras.layers.core import Dense

my_perceptron = Sequential ()

input_layer = Dense ( 1, input_dim=2 , activation="sigmoid" , kernel_initializer="zero" ) # 2 input for find 1 output using sigmoid function

# 1 is number of neurons in output

# https://en.wikipedia.org/wiki/Sigmoid_function is like any other functions, it is like f(x) = x+4 = y
# sigmoid is used for classification 0|1 in this case, from any inputs to 0|1 output

# Initializers define the way to set the initial weights of Keras layers --> kernel_initializer

my_perceptron.add ( input_layer )

from tensorflow.keras.optimizers import SGD

my_perceptron.compile ( loss="mse" , optimizer=SGD ( learning_rate=0.01 ) )

# the loss function determines the sum of the wrong trained values with an optimizer was chosen for perform a gradient Descent, the Derivate function

# mse! you must remember R, sum ( ( xi_n - midx_n )^2 ) / n

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

#### Creating the dataset

# mean and standard deviation for the x belonging to the first class
mu_x1, sigma_x1 = 0, 0.1

# constat to make the second distribution different from the first
x2_mu_diff = 0.35

import numpy  as np
import pandas as pd

# creating the first distribution
d1 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1 , 1000),
                   'x2': np.random.normal(mu_x1, sigma_x1 , 1000),
                   'type': 0})

# creating the second distribution
d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1 , 1000) + x2_mu_diff,
                   'x2': np.random.normal(mu_x1, sigma_x1 , 1000) + x2_mu_diff,
                   'type': 1})

data = pd.concat([d1, d2], ignore_index=True)

# Splitting the dataset in training and test set
msk = np.random.rand(len(data)) < 0.8

# Roughly 80% of data will go in the training set
train_x, train_y = data[['x1','x2']][msk], data.type[msk]

# Everything else will go into the valitation set
test_x, test_y = data[['x1','x2']][~msk], data.type[~msk]

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

my_perceptron.fit ( train_x.values , train_y , epochs=2 , batch_size=32 , shuffle=True ) # shuffle +- == random

from sklearn.metrics import roc_auc_score

pred_y = my_perceptron.predict ( test_x )

print ( roc_auc_score ( test_y, pred_y ) )

# perd : train_inputData = test : inputData

# some activation output functions are linear, A way to introduce non-linearity is by changing the activation function.
# such as Rectified Linear Unit (ReLU)
# sigmoid is non-linear function
