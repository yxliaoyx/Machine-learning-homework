import tensorflow as tf
import numpy as np
import time
from PIL import Image

start_time = time.time()

images = []
labels = []
one_hot_labels = []

for i in range(10):
    with open('FDDB-folds/FDDB-folds/FDDB-fold-%02d-ellipseList.txt' % (i + 1)) as annotation_file:
        for line in iter(annotation_file.readline, ''):
            image_file_path = line[:-1] + '.jpg'
            image_file = np.array(
                Image.open('originalPics/%s' % (image_file_path)).convert('RGB').resize((112, 112))) / 255
            number_of_faces = int(annotation_file.readline())
            if number_of_faces < 10:
                images.append(image_file)
                labels.append(number_of_faces - 1)
            for i in range(number_of_faces):
                annotation_file.readline()

for label in labels:
    one_hot_labels.append(np.eye(10)[label])

x = tf.placeholder("float", shape=[None, 112, 112, 3])
y_ = tf.placeholder("float", shape=[None, 10])

h_conv_1 = tf.layers.conv2d(x, 16, 5, 1, 'same', activation=tf.nn.relu)
h_pool_1 = tf.layers.max_pooling2d(h_conv_1, 2, 2)

h_conv0 = tf.layers.conv2d(h_pool_1, 16, 5, 1, 'same', activation=tf.nn.relu)
h_pool0 = tf.layers.max_pooling2d(h_conv0, 2, 2)

h_conv1 = tf.layers.conv2d(h_pool0, 32, 5, 1, 'same', activation=tf.nn.relu)
h_pool1 = tf.layers.max_pooling2d(h_conv1, 2, 2)

h_conv2 = tf.layers.conv2d(h_pool1, 64, 5, 1, 'same', activation=tf.nn.relu)
h_pool2 = tf.layers.max_pooling2d(h_conv2, 2, 2)

h_fc1 = tf.layers.dense(tf.reshape(h_pool2, [-1, 7 * 7 * 64]), 1024, tf.nn.relu)

y_conv = tf.layers.dense(h_fc1, 10, tf.nn.softmax)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

number_of_train_images = len(images) // 2
batch_size = 50
for i in range(100):
    for j in range(0, number_of_train_images, batch_size):
        batch_x = images[j:j + batch_size]
        batch_y = one_hot_labels[j:j + batch_size]
        if i % 1 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            print(time.time() - start_time, '秒')
        train_step.run(feed_dict={x: batch_x, y_: batch_y})

print("test accuracy %g" % accuracy.eval(
    feed_dict={x: images[number_of_train_images:], y_: one_hot_labels[number_of_train_images:]}))
