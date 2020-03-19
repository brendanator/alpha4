from consts import *
import tensorflow as tf
from tensorflow.python.client import device_lib
import util


class BaseNetwork(object):
    def __init__(self, scope, use_symmetry):
        self.scope = scope

        with tf.name_scope("inputs"):
            self.turn = tf.placeholder(tf.float32, shape=[None], name="turn")
            tiled_turn = tf.tile(
                tf.reshape(util.turn_win(self.turn), [-1, 1, 1, 1]),
                [1, 2, HEIGHT, WIDTH],
            )

            self.disks = tf.placeholder(
                tf.float32, shape=[None, 2, HEIGHT, WIDTH], name="disks"
            )

            self.empty = tf.placeholder(
                tf.float32, shape=[None, HEIGHT, WIDTH], name="empty"
            )
            empty = tf.expand_dims(self.empty, axis=1)

            self.legal_moves = tf.placeholder(
                tf.float32, shape=[None, HEIGHT, WIDTH], name="legal_moves"
            )
            legal_moves = tf.expand_dims(self.legal_moves, axis=1)

            self.threats = tf.placeholder(
                tf.float32, shape=[None, 2, HEIGHT, WIDTH], name="threats"
            )

            constant_features = np.array(
                [TILED_ROWS, ODDS, ROW_EDGE_DISTANCE, COLUMN_EDGE_DISTANCE],
                dtype=np.float32,
            ).reshape([1, 4, HEIGHT, WIDTH])
            batch_size = tf.shape(self.turn)[0]
            tiled_constant_features = tf.tile(constant_features, [batch_size, 1, 1, 1])

            feature_planes = tf.concat(
                [
                    tiled_turn,
                    self.disks,
                    empty,
                    legal_moves,
                    self.threats,
                    tiled_constant_features,
                ],
                axis=1,
            )

            if use_symmetry:
                # Interleave horizontally flipped position
                feature_planes_shape = [-1] + feature_planes.shape.as_list()[1:]
                flipped = tf.reverse(feature_planes, axis=[3])
                feature_planes = tf.reshape(
                    tf.stack([feature_planes, flipped], axis=1), feature_planes_shape
                )

        with tf.name_scope("conv_layers"):
            if self.gpu_available():
                data_format = "channels_first"
            else:
                feature_planes = tf.transpose(feature_planes, [0, 2, 3, 1])
                data_format = "channels_last"

            conv1 = tf.layers.conv2d(
                feature_planes,
                filters=32,
                kernel_size=[4, 5],
                padding="same",
                data_format=data_format,
                use_bias=False,
                name="conv1",
            )

            conv2 = tf.layers.conv2d(
                conv1,
                filters=32,
                kernel_size=[4, 5],
                padding="same",
                data_format=data_format,
                activation=tf.nn.relu,
                name="conv2",
            )

            conv3 = tf.layers.conv2d(
                conv2,
                filters=32,
                kernel_size=[4, 5],
                padding="same",
                data_format=data_format,
                activation=tf.nn.relu,
                name="conv3",
            )

            final_conv = tf.layers.conv2d(
                conv3,
                filters=1,
                kernel_size=[1, 1],
                data_format=data_format,
                name="final_conv",
            )
            disk_bias = tf.get_variable("disk_bias", shape=[TOTAL_DISKS])
            self.conv_output = tf.add(
                tf.contrib.layers.flatten(final_conv), disk_bias, name="conv_output"
            )

            self.conv_layers = [conv1, conv2, conv3, self.conv_output]

    def gpu_available(self):
        devices = device_lib.list_local_devices()
        return len([d for d in devices if d.device_type == "GPU"]) > 0

    @property
    def variables(self):
        # Add '/' to stop network-1 containing network-10 variables
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + "/")

    def assign(self, other):
        return [
            tf.assign(other_var, self_var)
            for self_var, other_var in zip(self.variables, other.variables)
        ]


class PolicyNetwork(BaseNetwork):
    def __init__(self, scope, temperature=1.0, reuse=None, use_symmetry=False):
        with tf.variable_scope(scope, reuse=reuse):
            super(PolicyNetwork, self).__init__(scope, use_symmetry)

            with tf.name_scope("policy"):
                self.temperature = tf.placeholder_with_default(
                    temperature, (), name="temperature"
                )

                disk_logits = tf.divide(
                    self.conv_output, self.temperature, name="disk_logits"
                )

                if use_symmetry:
                    # Calculate average of actual and horizontally flipped position
                    normal, flipped = tf.split(
                        tf.reshape(disk_logits, [-1, 2, HEIGHT, WIDTH]),
                        num_or_size_splits=2,
                        axis=1,
                    )
                    disk_logits = tf.reshape(
                        tf.reduce_mean(
                            tf.concat([normal, tf.reverse(flipped, axis=[3])], axis=1),
                            axis=1,
                        ),
                        [-1, TOTAL_DISKS],
                    )

                # Make illegal moves impossible:
                #   - Legal moves have positive logits
                #   - Illegal moves have -ILLEGAL_PENALTY logits
                legal_moves = tf.contrib.layers.flatten(self.legal_moves)
                legal_disk_logits = (
                    tf.nn.relu(disk_logits) * legal_moves
                    + (legal_moves - 1) * ILLEGAL_PENALTY
                )

                self.policy = tf.nn.softmax(legal_disk_logits, name="policy")
                self.sample_move = tf.squeeze(
                    tf.multinomial(legal_disk_logits, 1) % WIDTH,
                    axis=1,
                    name="sample_move",
                )

                self.entropy = tf.reduce_sum(
                    self.policy * -tf.log(self.policy + EPSILON),  # Avoid Nans
                    axis=1,
                    name="entropy",
                )

                self.policy_layers = self.conv_layers + [
                    disk_logits,
                    self.policy,
                    self.entropy,
                ]


class ValueNetwork(BaseNetwork):
    def __init__(self, scope, use_symmetry=False):
        with tf.variable_scope(scope):
            super(ValueNetwork, self).__init__(scope, use_symmetry)

            with tf.name_scope("value"):
                fully_connected = tf.layers.dense(
                    self.conv_output,
                    units=64,
                    activation=tf.nn.relu,
                    name="fully_connected",
                )

                value = tf.layers.dense(fully_connected, 1, tf.tanh)

                if use_symmetry:
                    # Calculate average of actual and horizontally flipped position
                    self.value = tf.reduce_mean(
                        tf.reshape(value, [-1, 2]), axis=1, name="value"
                    )
                else:
                    self.value = tf.squeeze(value, axis=1, name="value")

                self.value_layers = self.conv_layers + [fully_connected, self.value]
