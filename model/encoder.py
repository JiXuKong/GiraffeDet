import tensorflow as tf

from model.layers import conv2d, bn_, act

class DilatedEncoder(object):
    def __init__(self, encoder_channels,
        block_mid_channels,
        num_residual_blocks,
        block_dilations,
        is_training):
        self.encoder_channels = encoder_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = block_dilations
        self.is_training = is_training
        assert len(self.block_dilations) == self.num_residual_blocks

    def build_layers(self, x):
        x = conv2d(x, self.encoder_channels,
                                      kernel=1,
                                      is_training=self.is_training,
                                      scope='LateralConv')
        x = bn_(x, is_training=self.is_training, scope='LateralNnorm')
        x = conv2d(x, self.encoder_channels,
                                  kernel=3,
                                  is_training=self.is_training,
                                  scope='FpnConv')
        x = bn_(x, is_training=self.is_training, scope='FpnNorm')

        for i in range(self.num_residual_blocks):
            x = self.Bottleneck(x, self.block_dilations[i], is_training=self.is_training, scope=i)
        return x


    def Bottleneck(self, x, rate, is_training, scope):
        with tf.variable_scope('DilationBottleneck' + str(scope)):
            identity = x
            x = conv2d(x, self.block_mid_channels, kernel=1, is_training=is_training, scope='Conv1')
            x = bn_(x, is_training=is_training, scope='Conv1Norm')
            x = act(x)
            x = conv2d(x, self.block_mid_channels, kernel=3, dilation=rate, is_training=is_training, scope='Conv2')
            x = bn_(x, is_training=is_training, scope='Conv2Norm')
            x = act(x)
            x = conv2d(x, self.encoder_channels, kernel=1, is_training=is_training, scope='Conv3')
            x = bn_(x, is_training=is_training, scope='Conv3Norm')
            x = act(x)
            return x + identity