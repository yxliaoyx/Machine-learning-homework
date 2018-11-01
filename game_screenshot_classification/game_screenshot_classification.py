import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

MODEL_DIR = 'model/'
MODEL_FILE = 'tensorflow_inception_graph.pb'
CACHE_DIR = 'data/tmp/bottleneck'
INPUT_DATA_DIR = 'data/screenshots'
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
BOTTLENECK_TENSOR_SIZE = 2048
LEARNING_RATE = 0.01
STEPS = 100
BATCH = 100
CHECKPOINT_EVERY = 100


def create_image_lists(validation_percentage, test_percentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA_DIR)]
    is_root_dir = True

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        file_list = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(INPUT_DATA_DIR, dir_name, '*.jpeg')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            continue

        label_name = dir_name
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (validation_percentage + test_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }
    return result


def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    sub_dir_path = os.path.join(CACHE_DIR, label_name)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = os.path.join(CACHE_DIR, label_name, image_lists[label_name][category][index])

    if not os.path.exists(bottleneck_path):
        image_path = os.path.join(INPUT_DATA_DIR, label_name, image_lists[label_name][category][index])
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = np.squeeze(sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data}))

        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randint(0, len(image_lists[label_name][category]) - 1)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category, jpeg_data_tensor,
                                              bottleneck_tensor)
        bottlenecks.append(bottleneck)
        ground_truths.append(np.eye(n_classes)[label_index])
    return bottlenecks, ground_truths


def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for label_index, label_name in enumerate(list(image_lists.keys())):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor,
                                                  bottleneck_tensor)
            bottlenecks.append(bottleneck)
            ground_truths.append(np.eye(n_classes)[label_index])
    return bottlenecks, ground_truths


def main(_):
    image_lists = create_image_lists(VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    n_classes = len(image_lists.keys())

    with tf.Graph().as_default() as graph:
        with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def,
                                                                      return_elements=['pool_3/_reshape:0',
                                                                                       'DecodeJpeg/contents:0'])
        bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

        with tf.name_scope('final_training_ops'):
            final_tensor = tf.layers.dense(bottleneck_input, n_classes)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=final_tensor, labels=ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        import time
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', str(int(time.time()))))
        print('Writing to {}'.format(out_dir))

        loss_summary = tf.summary.scalar('loss', cross_entropy_mean)
        acc_summary = tf.summary.scalar('accuracy', evaluation_step)
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH,
                                                                                  'training', jpeg_data_tensor,
                                                                                  bottleneck_tensor)
            _, train_summaries = sess.run([train_step, train_summary_op],
                                          feed_dict={bottleneck_input: train_bottlenecks,
                                                     ground_truth_input: train_ground_truth})
            train_summary_writer.add_summary(train_summaries, i)

            if i % 100 == 0 or i == STEPS - 1:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess, n_classes,
                                                                                                image_lists, BATCH,
                                                                                                'validation',
                                                                                                jpeg_data_tensor,
                                                                                                bottleneck_tensor)
                validation_accuracy, dev_summaries = sess.run([evaluation_step, dev_summary_op],
                                                              feed_dict={bottleneck_input: validation_bottlenecks,
                                                                         ground_truth_input: validation_ground_truth})
                print('Step %d : Validation accuracy on random sampled %d examples = %.1f%%' % (
                    i, BATCH, validation_accuracy * 100))

            if i % CHECKPOINT_EVERY == 0:
                dev_summary_writer.add_summary(dev_summaries, i)
                path = saver.save(sess, checkpoint_prefix, global_step=i)
                print('Saved model checkpoint to {}\n'.format(path))

        test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor,
                                                                   bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step,
                                 feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

        output_labels = os.path.join(out_dir, 'labels.txt')
        with tf.gfile.FastGFile(output_labels, 'w') as f:
            keys = list(image_lists.keys())
            for i in range(len(keys)):
                keys[i] = '%2d -> %s' % (i, keys[i])
            f.write('\n'.join(keys) + '\n')


if __name__ == '__main__':
    tf.app.run()
