import tensorflow as tf
import numpy as np
import os

import data_helper
from tensorflow.contrib import learn
import csv

#
tf.flags.DEFINE_boolean("eval_train", False, "Evaluation on training data")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1533479027/checkpoints", "checkpoint directory from training run")

FLAGS = tf.flags.FLAGS

if FLAGS.eval_train:
    sentence_1, sentence_2, y = data_helper.load_data("data/SICK_data.txt")
else:
    sentence_1 = ["A group of kids is playing in a yard and an old man is standing in the background",
                  "A group of children is playing in the house and there is no man standing in the background"]
    sentence_2 = ["A group of boys in a yard is playing and a man is standing in the background",
                  "A group of kids is playing in a yard and an old man is standing in the background"]


# Map data into vocabularies
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
sentence_test_1 = np.array(list(vocab_processor.transform(sentence_1)))
sentence_test_2 = np.array(list(vocab_processor.transform(sentence_2)))
y = [4]

print("\nEvaluation...\n")

# Evaluation
checkpoint_dir = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()

with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_dir))
        saver.restore(sess, checkpoint_dir)

        # Get the placeholders from the graph by name
        input_1 = graph.get_operation_by_name("input_1").outputs[0]
        input_2 = graph.get_operation_by_name("input_2").outputs[0]

        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("Dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/scores").outputs[0]

        # batches generating
        #data = list(zip(sentence_test_1, sentence_test_2, y))
        #batches = data_helper.batches_generate(data, 2, 2)

        #for batch in batches:
        #    sentence_batch_1, sentence_batch_2, y = zip(*batch)
        #    feed_dict = {input_1: sentence_batch_1, input_2: sentence_batch_2, dropout_keep_prob: 1.0}
        #    tmp = sess.run(predictions, feed_dict)
        #    result.append(tmp)
        feed_dict = {input_1: sentence_test_1, input_2: sentence_test_2, dropout_keep_prob: 1.0}
        result = sess.run(predictions, feed_dict)

for i in range(len(sentence_1)):
    print("\n{}\n{}\n{}".format(sentence_1[i], sentence_2[i], result[i]))