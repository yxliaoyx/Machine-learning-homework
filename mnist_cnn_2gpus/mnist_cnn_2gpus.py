import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

start_time = time.time()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

split_x = tf.split(tf.reshape(x, [-1, 28, 28, 1]), 2)
split_y_ = tf.split(y_, 2)

correct_prediction = []
losses = []

for gpu_id in range(2):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            h_conv1 = tf.layers.conv2d(split_x[gpu_id], 32, 5, 1, 'same', activation=tf.nn.relu)
            h_pool1 = tf.layers.max_pooling2d(h_conv1, 2, 2)

            h_conv2 = tf.layers.conv2d(h_pool1, 64, 5, 1, 'same', activation=tf.nn.relu)
            h_pool2 = tf.layers.max_pooling2d(h_conv2, 2, 2)

            h_fc1 = tf.layers.dense(tf.reshape(h_pool2, [-1, 7 * 7 * 64]), 1024, tf.nn.relu)

            y_conv = tf.layers.dense(h_fc1, 10)
            correct_prediction.append(tf.equal(tf.argmax(y_conv, 1), tf.argmax(split_y_[gpu_id], 1)))

            cost = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=split_y_[gpu_id])
            losses.append(cost)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss, colocate_gradients_with_ops=True)

init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(init)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 20000 * 50
for i in range(100):
    batch = mnist.train.next_batch(10000)
    train_loss, _ = sess.run([loss, optimizer], feed_dict={x: batch[0], y_: batch[1]})

    if i % 10 == 0:
        print("step %d, loss %g" % (i, train_loss))
        print(time.time() - start_time, '秒')

print("test accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(time.time() - start_time, '秒')
