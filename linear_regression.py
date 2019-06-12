import numpy as np 
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

housing = fetch_california_housing()
m, n = housing.data.shape
#print(m)
#print(n)
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
#print(housing_data_plus_bias[1:5,:])
X = tf.constant(housing_data_plus_bias,dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name="y")
#normal equation calculation
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)

with tf.Session() as sess:
    theta_value = theta.eval()
#print(theta_value)

#implementing gradient descent
#gradient descent requires us to normalize the input feature vectors 
#this can be done using numpy, tensorflow or scikit learn standardscaler
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
housing_scaled_data = sc.fit_transform(housing_data_plus_bias)

n_epochs = 1000
learning_rate = 0.01

x_train = tf.constant(housing_scaled_data,dtype=tf.float32, name="x_train")
y_train = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32, name="y_train")

theta = tf.Variable(tf.random_uniform([n+1,1],-1,1.0),name="theta")
y_pred = tf.matmul(x_train, theta, name="predictions")

error = y_pred - y_train

mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(x_train),error)

training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
print(best_theta)