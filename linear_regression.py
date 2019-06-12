import numpy as np 
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

housing = fetch_california_housing()
m, n = housing.data.shape
#print(m)
#print(n)
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
#print(housing_data_plus_bias[1:5,:])
X = tf.constant(housing_data_plus_bias, name="X")
y = tf.constant(housing.target.reshape(-1,1),name="y")
#normal equation calculation
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)

with tf.Session() as sess:
    theta_value = theta.eval()
print(theta_value)

#implementing gradient descent
#gradient descent requires us to normalize the input feature vectors 
#this can be done using numpy, tensorflow or scikit learn standardscaler
