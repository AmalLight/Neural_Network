import numpy as np
np.random.seed ( 11 )

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def sigmoid ( s ) : # Activation function
    return 1 / (1 + np.exp(-s))          # input = AxB --> Output = 1/(1+pi^(-(AxB))) = AxB, Originally pi was the neperus number

def sigmoid_prime ( s ) : # Derivative of the sigmoid
    return sigmoid(s) * (1 - sigmoid(s)) # input = AxB --> Output = AxB * 1-AxB = AxB | * is not dot

# -----------------------------------------------------------------------------

class FFNN(object):

    def __init__(self, input_size=2, hidden_size=2, output_size=1):

        # Adding 1 as it will be our bias

       self.input_size = input_size + 1
       self.hidden_size = hidden_size + 1
       self.output_size = output_size
    
       self.o_error = 0
       self.o_delta = 0

       self.z1 = 0
       self.z2 = 0
       self.z3 = 0
       self.z2_error = 0

       # The whole weight matrix, from the inputs till the hidden layer
       # The final set of weights from the hidden layer till the output layer

               # generates matrix #  set for rows    #  set for columns
       self.w1 = np.random.randn  ( self.input_size  , self.hidden_size ) # (I+1)x(H+1)   = 3x3

       self.w2 = np.random.randn  ( self.hidden_size , self.output_size ) #       (H+1)xO = 3x1
        
    def forward(self, X):

                      # Forward propagation through our network
        X['bias'] = 1 # Adding 1 to the inputs to include the bias in the weight
                      # => 3200x3 | X = 3200x2
 
        self.z1 = np.dot  ( X , self.w1 )          # 3200x3 * 3x3 = 3200x3 | * is dot
                                                   # Here w1 using bias ( implicit )

        self.z2 = sigmoid (        self.z1       ) # activation function => 3200x3
        # print ( 'shape of z2:' , self.z1.shape ) # I had right         => 3200x3
        
        self.z3 = np.dot  (    self.z2 , self.w2 ) # dot product of hidden layer (z2) and second set of 3x1 weights
                                                   # 3200x3 * 3x1 = 3200x1 | * is dot
                                                   # Here w2 using bias ( implicit )

        out = sigmoid ( self.z3 ) # final activation function = 3200x1
        return out
       
    def predict(self, X):
        return forward(self, X)
       
    def backward(self, X, y, output, step): # Using only Derivate of Activation and pay for 2 executions, And remove Loss+Gradient_Descent functions

        # Backward propagation of the errors
        X['bias'] = 1 # Adding 1 to the inputs to include the bias in the weight

        self.o_error = y - output # error in output => 3200x1

        self.o_delta = self.o_error * sigmoid_prime(output) * step # applying derivative of sigmoid to error => 3200x1 * 3200x1 = 3200x1 | * is not dot
       
        self.z2_error = self.o_delta.dot ( self.w2.T ) # z2 error: how much our hidden layer weights contributed to output error => 3200x1 * 1x3 = 3200x3 | * is dot

        self.z2_delta = self.z2_error * sigmoid_prime(self.z2) * step # applying derivative of sigmoid to z2 error => 3200x3 * 3200x3 * 0.5 = 3200x3 | all * is not dot
       
        self.w1 += X.T.dot(self.z2_delta)      # adjusting first      of weights => 3x3200 * 3200x3 => 3x3 => W1 += 3x3 = 3x3 | * is dot

        self.w2 += self.z2.T.dot(self.o_delta) # adjusting second set of weights => 3x3200 * 3200x1 => 3x1 => W2 += 3x1 = 3x1 | * is dot

    def fit(self, X, y, epochs=10, step=0.05):

        for epoch in range(epochs): # epoch with Backward propagation costs only 2 * ( Matrix*W ), with GPU Matrix*W is faster

            X['bias'] = 1 # Adding 1 to the inputs to include the bias in the weight => 3200x3 | X = 3200x2

            output = self.forward(X)

            self.backward(X, y, output, step)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#### Creating the dataset
# mean and standard deviation for the x belonging to the first class

mu_x1, sigma_x1 = 0, 0.1

# Constant to make the second distribution different from the first

x1_mu_diff, x2_mu_diff, x3_mu_diff, x4_mu_diff = 0, 1, 0, 1

# x1_mu_diff, x2_mu_diff, x3_mu_diff, x4_mu_diff = 0.5, 0.5, 0.5, 0.5

# creating the first distribution

import pandas as pd

                            # generates vect # start #   end    #  set
d1 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) + 0,
                      'x2': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) + 0, 'type': 0 } )

d2 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) + 1,
                      'x2': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) - 0, 'type': 1 } )

d3 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) - 0,
                      'x2': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) - 1, 'type': 0 } )

d4 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) - 1,
                      'x2': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) + 1, 'type': 1 } )

# variables are columns, rows are the datasets and each row is one completed input, you must remember R

data = pd.concat ( [d1, d2, d3, d4] , ignore_index=True ) # concat of rows

print ( 'print len data row:' , len ( data             ) ) # 4000
print ( 'print len data col:' ,       data.shape [ 1 ] )   # 3, third is type

print ( 'example x1: ', data [  'x1'  ] [ 5 ] ) # 0.
print ( 'example x2: ', data [  'x2'  ] [ 5 ] ) # 0.
print ( 'example  3: ', data [ 'type' ] [ 5 ] ) # 0

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

# Splitting the dataset in training and test set
msk = np.random.rand(len(data)) < 0.8 # test will be +- 20% == +- 800

# Roughly 80% of data will go in the training set
train_x, train_y = data[['x1', 'x2']][msk], data[['type']][msk].values # train will be +- 80% == +- 3200

# Everything else will go into the validation set
test_x, test_y = data[['x1', 'x2']][~msk], data[['type']][~msk].values

my_network = FFNN()

# my_network.fit(train_x, train_y, epochs=10, step=0.001)
my_network.fit(train_x, train_y, epochs=10000, step=0.001)

pred_y = test_x.apply(my_network.forward, axis=1)

# Reshaping the data
test_y_ = [i[0] for i in test_y]
pred_y_ = [i[0] for i in pred_y]

from sklearn.metrics import roc_auc_score , mean_squared_error , confusion_matrix

print('MSE: ', mean_squared_error(test_y_, pred_y_))
print('AUC: ', roc_auc_score(test_y_, pred_y_))

threshold = 0.5

pred_y_binary = [0 if i > threshold else 1 for i in pred_y_]

cm = confusion_matrix(test_y_, pred_y_binary, labels=[0, 1])

print ( pd.DataFrame ( cm , index = [ 'True 0', 'True 1' ] , columns = [ 'Predicted 0' , 'Predicted 1' ] ) )

