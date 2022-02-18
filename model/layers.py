import tensorflow as tf

slim = tf.contrib.slim

def act(x):
    return tf.nn.relu6(x)

def SiLU(x, beta=1.0, scope='SiLU'):
    with tf.name_scope(scope):
        return x*tf.nn.sigmoid(beta * x)


def conv2d(input, channel, kernel, stride=1, dilation=1, reuse=False, scope=None, bias = None, is_training=True):
        output = slim.conv2d(input, channel, kernel,
                                    weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                    stride=stride,
                                    rate = dilation,
                                    padding="SAME",
                                    biases_initializer=bias,
                                    activation_fn=None,
                                    reuse = reuse,
                                    trainable=is_training,
                                    scope=scope)
        return output


def bn_(input_, esp=1e-3, is_training = True, decay = 0.99, scope = 'bn'):
    # x = tf.layers.batch_normalization(
    #     inputs = input_,
    #     axis=-1,
    #     name = scope,
    #     momentum= 0.997,
    #     epsilon= 1e-4,
    #     training= is_training)
        # fused=True)
    x= slim.batch_norm(
        inputs=input_,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        is_training=is_training,
        scope=scope)
    return x


# PixelShuffler layer for Keras
# by t-ae
# https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981
def Space_to_depth(x, size=(2, 2), scope='None'):
    with tf.name_scope(scope):
        batch_size, h, w, c = tf.shape(x)

        rh, rw = size
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)

        out = tf.reshape(x, (batch_size, h, w, rh, rw, oc))
        out = tf.transpose(out, (0, 1, 3, 2, 4, 5))
        out = tf.reshape(out, (batch_size, oh, ow, oc))
        return out


def upsample_layer(inputs, out_shape, scope):
    with tf.name_scope(scope):
        new_height, new_width = out_shape[0], out_shape[1]

        channels = inputs.shape[3].value
        rate = 2
        x = tf.reshape(inputs, [-1, new_height//rate, 1, new_width//rate, 1, channels])
        x = tf.tile(x, [1, 1, rate, 1, rate, 1])
        x = tf.reshape(x, [-1, new_height, new_width, channels])
        return x

def downsample_layer(inputs, scope):
    with tf.name_scope(scope):
        x = slim.max_pool2d(inputs, [2, 2], stride=2, padding='SAME')
        return x