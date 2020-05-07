import math
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

tf.random.set_random_seed(1950)
np.random.seed(1950)

MODEL_SAVER_PATH = "save"


def fitness_func(loss, flops, T=10):
    return loss * (flops / T) ** 0.06


class PoseTrainer:
    def __init__(self, input_placeholder, output_placeholder, training_placeholder, session,
                 export_dir=None, verbose=False, name=None):
        self.session = session
        self.input = input_placeholder
        self.output = output_placeholder
        self.is_training = training_placeholder
        self.name = name
        self.export_dir = export_dir
        self.validation_loss = None
        self.flops = None
        self.__build_training()

        if export_dir: self.restore_model(export_dir)
        if verbose: print("\n".join([n.name for n in tf.get_default_graph().as_graph_def().node]))
        if not name: self.name = datetime.now().strftime("model_%d_%m_%Y_%H_%M")

    def __build_training(self):
        self.angle = tf.placeholder(shape=(None, 3), dtype=tf.float32)
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[])

        self.l2_loss = tf.losses.get_regularization_loss()
        self.loss_function = tf.compat.v1.losses.huber_loss(labels=self.angle, predictions=self.output, delta=5.0)
        self.loss_total = self.loss_function + self.l2_loss

        self.mae = tf.abs(tf.subtract(self.angle, self.output))
        self.mean_mae = tf.reduce_mean(tf.abs(tf.subtract(self.angle, self.output)))

        self.optimizer = tf.train.AdamOptimizer(self.lr_placeholder).minimize(self.loss_total)
        self.update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.optimizer = tf.group([self.update_ops, self.optimizer])

    def restore_model(self, export_dir):
        print("Restoring model...")
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.session, export_dir)
        print("Successfully restored.")

    def save_model(self):
        print("Restoring model...")
        saver = tf.train.Saver(tf.global_variables())
        saver.save(self.session, f"{MODEL_SAVER_PATH}/{self.name}/{self.name}.ckpt")
        print("Successfully saved.")

    def test_forward(self, dataset, training_node=True):
        dataset.initialize(self.session)
        batches_per_epoch = dataset.batches_per_epoch()
        progress_bar = tqdm(total=batches_per_epoch)

        mae = []
        while True:
            image, angle = dataset.get_batch(self.session)
            if image is None: break

            if training_node:
                res = self.session.run(self.mae, feed_dict={
                    self.input: image,
                    self.angle: angle,
                    self.is_training: False
                })
            else:
                res = self.session.run(self.mae, feed_dict={
                    self.input: image,
                    self.angle: angle
                })
            mae.append(res)
            progress_bar.update(1)

        res = np.concatenate(mae, axis=0).mean(axis=0)
        return res

    def train(self, dataset, dataset_val, dataset_test=None, epochs=24, learning_rate=[0.1],
              epoch_print=True, multi=True):
        self.learning_rate = learning_rate
        lr_epochs = math.ceil(epochs / len(self.learning_rate))
        lr_ = 0

        train_loss_acc, val_loss_acc, test_loss_acc = [], [], []
        self.best_val = None
        for epoch in range(epochs):
            if epoch_print: print('{}: {}.'.format('EPOCH'.ljust(12), epoch + 1))

            # Datasets initialization
            dataset.initialize(self.session)
            dataset_val.initialize(self.session)
            if dataset_test:  dataset_test.initialize(self.session)

            # Training set update
            train_loss, train_batches = self._dataset_pass(dataset,
                                                           learning_rate=self.learning_rate[lr_],
                                                           gradient=True)
            self.train_loss = train_loss / train_batches

            # Validation set update
            validation_loss, val_batches = self._dataset_pass(dataset_val,
                                                              gradient=False)
            self.validation_loss = validation_loss / val_batches

            # Test set update
            if dataset_test:
                test_loss, test_batches = self._dataset_pass(dataset_test,
                                                             gradient=False)

            train_loss_acc.append(train_loss / train_batches)
            val_loss_acc.append(validation_loss / val_batches)

            if self.best_val is None or self.validation_loss < self.best_val:
                self.save_model()
                self.freeze_model()
                self.best_val = self.validation_loss

            if epoch_print:
                print('{}:{:0.6f}'.format('epoch_loss'.ljust(12), train_loss / train_batches))
                print('{}:{:0.6f}'.format('val_loss'.ljust(12), validation_loss / val_batches))
                if dataset_test: print('{}:{:0.6f}'.format('test_loss'.ljust(12), test_loss / test_batches))
                print()

            if (epoch + 1) % lr_epochs == 0 and lr_epochs != 0:
                lr_ += 1
                if lr_ >= len(self.learning_rate): continue
                print("Changing learning rate: {}".format(self.learning_rate[lr_]))

        if multi:
            self.fitness = fitness_func(self.best_val, self.flops)
            return self.fitness
        else:
            return self.best_val

    def _train_forward(self, input_tensor, angle_tensor, learning_rate=None, gradient=True):
        if gradient:
            _, loss_val = self.session.run([self.optimizer, self.loss_function],
                                           feed_dict={
                                               self.input: input_tensor,
                                               self.angle: angle_tensor,
                                               self.lr_placeholder: learning_rate,
                                               self.is_training: gradient
                                           })

        else:
            loss_val = self.session.run(self.mean_mae,
                                        feed_dict={
                                            self.input: input_tensor,
                                            self.angle: angle_tensor,
                                            self.is_training: gradient
                                        })
        return loss_val

    def _dataset_pass(self, dataset, learning_rate=None, gradient=True):
        epoch_loss, num_batches = 0, 0
        batches_per_epoch = dataset.batches_per_epoch()
        progress_bar = tqdm(total=batches_per_epoch)
        while True:
            image, angles = dataset.get_batch(self.session)
            if image is None: break

            loss_val = self._train_forward(image, angles,
                                           learning_rate=learning_rate,
                                           gradient=gradient)

            epoch_loss += loss_val
            num_batches += 1
            progress_bar.set_description('{}:{:0.4f}'.format('batch_loss'.ljust(12),
                                                             epoch_loss / num_batches))
            progress_bar.update(1)

        progress_bar.close()
        return epoch_loss, num_batches

    def freeze_model(self):
        from tensorflow.python.framework import graph_util
        proto_path = f"{MODEL_SAVER_PATH}/{self.name}/"
        proto_name = f"{self.name}.pb"

        gd = self.session.graph.as_graph_def()

        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        converted_graph_def = graph_util.convert_variables_to_constants(self.session, gd, ["pose_output"])
        tf.train.write_graph(converted_graph_def, proto_path, proto_name, as_text=False)
        if not self.flops: self.set_flops()

    def set_flops(self):
        model_path = f"{MODEL_SAVER_PATH}/{self.name}/{self.name}.pb"

        def load_pb(pb):
            with tf.gfile.GFile(pb, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                input_map = {}
                shape = self.input.get_shape().as_list()
                shape[0] = 1
                input_map["pose_input"] = tf.ones(shape=shape,
                                                  dtype=tf.float32,
                                                  name="pose_input")

                input_map["is_training"] = tf.constant(False,
                                                       dtype=tf.bool,
                                                       name="is_training")

                tf.import_graph_def(graph_def, name='', input_map=input_map)
                return graph

        g = load_pb(model_path)
        with g.as_default():
            flops = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.float_operation())
            print(f'TOTAL_FLOPS: {flops.total_float_ops / 1_000_000} millions')

        self.flops = flops.total_float_ops / 1_000_000
