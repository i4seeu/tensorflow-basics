import numpy as np 
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

housing = fetch_california_housing()
m, n = housing.data.shape
#print(m)
#print(n)
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

#implementing gradient descent
#gradient descent requires us to normalize the input feature vectors 
#this can be done using numpy, tensorflow or scikit learn standardscaler
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
housing_scaled_data = sc.fit_transform(housing_data_plus_bias)

n_epochs = 1000
learning_rate = 0.01

#x_train = tf.constant(housing_scaled_data,dtype=tf.float32, name="x_train")
#y_train = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32, name="y_train")
x_train = tf.placeholder(tf.float32, shape=(None, n+1), name="x_train")
y_train = tf.placeholder(tf.float32, shape=(None, 1), name="y_train")

batch_size = 100
n_batches = int(np.ceil(m/batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    x_batch = housing_scaled_data[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return x_batch, y_batch

theta = tf.Variable(tf.random_uniform([n+1,1],-1,1.0),name="theta")
y_pred = tf.matmul(x_train, theta, name="predictions")

error = y_pred - y_train

mse = tf.reduce_mean(tf.square(error), name="mse")
#using an optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            x_batch, y_batch = fetch_batch(epoch,batch_index, batch_size)
            sess.run(training_op, feed_dict={x_train:x_batch, y_train:y_batch})
    best_theta = theta.eval()
print(best_theta)

#using autodiff
#tensoflow calculate the gradients automatically