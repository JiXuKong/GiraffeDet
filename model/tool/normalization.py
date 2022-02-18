import tensorflow as tf
def bn_(input_, esp=1e-3, is_training = True, decay = 0.99, scope = 'bn'):

    x = tf.layers.batch_normalization(
        inputs = input_,
        axis=-1,
        name = scope,
        momentum= 0.997,
        epsilon= 1e-4,
        training= is_training)
#         fused=True)
    return x