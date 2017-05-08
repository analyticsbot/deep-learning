import tensorflow as tf

x = tf.placeholder("float", [None, 3])
y = x ** 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3, 4]})
    print(result)
