import tensorflow as tf

#defining our variables using tensorflow
x = tf.Variable(3,name="x")
y = tf.Variable(4,name="y")

f = x*x*y+ y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

result = 0
#a better way to implement the above code is to use the global initialize
init = tf.global_variables_initializer() # prepare an init node
with tf.Session() as sess:
    init.run()
    result = f.eval()
print(result)