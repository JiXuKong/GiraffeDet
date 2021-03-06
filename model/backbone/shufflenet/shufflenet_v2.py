import tensorflow as tf
import tensorflow.contrib.slim as slim

DATA_FORMAT = 'channels_last'
BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3

@tf.contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        data_format='NHWC', scope='depthwise_conv'):
    with tf.variable_scope(scope):

        if data_format == 'NHWC':
            in_channels = x.shape[3].value
            strides = [1, stride, stride, 1]
        else:
            in_channels = x.shape[1].value
            strides = [1, 1, stride, stride]

        W = tf.get_variable(
            'depthwise_weights',
            [kernel, kernel, in_channels, 1],
            dtype=tf.float32
        )
        x = tf.nn.depthwise_conv2d(x, W, strides, padding, data_format=data_format)
        x = normalizer_fn(x) if normalizer_fn is not None else x  # batch normalization
        x = activation_fn(x) if activation_fn is not None else x  # nonlinearity
        return x

def shufflenet_v2(images, is_training, depth_multiplier='1.0'):
    """
    This is an implementation of ShuffleNet v2:
    https://arxiv.org/abs/1807.11164

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        depth_multiplier: a string, possible values are '0.5', '1.0', '1.5', and '2.0'.
    Returns:
        a dict with float tensors.
    """
    data_format = 'NCHW' if DATA_FORMAT == 'channels_first' else 'NHWC'

    possibilities = {'0.5': 48, '1.0': 116, '1.5': 176, '2.0': 224}
    initial_depth = possibilities[depth_multiplier]

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=1 if DATA_FORMAT == 'channels_first' else 3,
            center=True, scale=True,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            training=is_training, fused=True,
            name='batch_norm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    with tf.variable_scope('ShuffleNetV2'):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': data_format
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):
            features = {}

            if DATA_FORMAT == 'channels_first':
                x = tf.transpose(x, [0, 3, 1, 2])

            x = slim.conv2d(x, 24, (3, 3), stride=2, scope='Conv1')
            x = slim.max_pool2d(
                x, (3, 3), stride=2, padding='SAME',
                data_format=data_format, scope='MaxPool'
            )

            stage_name = 'Stage2'
            x = block(x, num_units=4, out_channels=initial_depth, scope=stage_name)
            features[stage_name] = x  # stride 8

            stage_name = 'Stage3'
            x = block(x, num_units=8, scope=stage_name)
            features[stage_name] = x  # stride 16

            stage_name = 'Stage4'
            x = block(x, num_units=4, scope=stage_name)
            features[stage_name] = x

            final_channels = 1024 if depth_multiplier != '2.0' else 2048
            x = slim.conv2d(x, final_channels, (1, 1), stride=1, scope='Conv5')
            features['Conv5'] = x  # stride 32

    return {
        'p3': features['Stage2'],
        'p4': features['Stage3'],
        'p5': features['Conv5']
    }


def block(x, num_units, out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(x, out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)
                x = basic_unit(x)
        x = tf.concat([x, y], axis=1 if DATA_FORMAT == 'channels_first' else 3)

    return x


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):

        shape = tf.shape(x)
        batch_size = x.shape[0].value
        if batch_size is None:
            batch_size = shape[0]

        if DATA_FORMAT == 'channels_first':
            depth, height, width = x.shape[1].value, shape[2], shape[3]
            z = tf.stack([x, y], axis=1)  # shape [batch_size, 2, depth, height, width]
            z = tf.transpose(z, [0, 2, 1, 3, 4])
            z = tf.reshape(z, [batch_size, 2*depth, height, width])
            x, y = tf.split(z, num_or_size_splits=2, axis=1)
        else:
            height, width, depth = shape[1], shape[2], x.shape[3].value
            z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]
            z = tf.transpose(z, [0, 1, 2, 4, 3])
            z = tf.reshape(z, [batch_size, height, width, 2*depth])
            x, y = tf.split(z, num_or_size_splits=2, axis=3)

        return x, y


def basic_unit(x):
    in_channels = x.shape[1].value if DATA_FORMAT == 'channels_first' else x.shape[3].value
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    x = depthwise_conv(x, kernel=3, stride=1, activation_fn=None, scope='depthwise')
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_after')
    return x


def basic_unit_with_downsampling(x, out_channels=None):
    in_channels = x.shape[1].value if DATA_FORMAT == 'channels_first' else x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    y = depthwise_conv(y, kernel=3, stride=2, activation_fn=None, scope='depthwise')
    y = slim.conv2d(y, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')

    with tf.variable_scope('second_branch'):
        x = depthwise_conv(x, kernel=3, stride=2, activation_fn=None, scope='depthwise')
        x = slim.conv2d(x, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')
        return x, y
