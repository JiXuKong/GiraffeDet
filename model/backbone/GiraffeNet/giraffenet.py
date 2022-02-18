import tensorflow as tf
from model.layers import conv2d, bn_, SiLU, Space_to_depth


def stem(x, dim, is_training, scope):
    with tf.variable_scope(scope):
        x = conv2d(x, dim, 3, stride=2, is_training=is_training, scope='3x3conv_1')
        x = bn_(x, is_training=is_training, scope="3x3conv_1BN")
        x = SiLU(x)
        x = conv2d(x, dim*2, 3, stride=2, is_training=is_training, scope='3x3conv_2')
        x = bn_(x, is_training=is_training, scope="3x3conv_2BN")
        x = SiLU(x)
        return x



def S2D_Block(x, dim, is_training, scope=None):
    '''
    Impletment S2D_Block
    Space-to-depth(2*dim)->Conv(dim)->SiLU(dim)
    '''
    with tf.variable_scope(scope):
        # x = Space_to_depth(x, scope='Space_to_depth')
        x = tf.nn.space_to_depth(x, 2)
        x = conv2d(x, dim, 1, is_training=is_training, scope='1x1conv')
        x = bn_(x, is_training=is_training, scope="1x1convBN")
        x = SiLU(x)
        return x


def S2DChain(x, dim, is_training, scope="S2DChain"):
    with tf.variable_scope(scope):
        x = stem(x, dim, is_training, 'stem')
        x = S2D_Block(x, 4*dim, is_training, 'S2D_Block1')
        dim = 4*dim
        P3 = x
        x = S2D_Block(x, dim, is_training, 'S2D_Block2')
        dim = 2*dim
        P4 = x
        x = S2D_Block(x, dim, is_training, 'S2D_Block3')
        dim = 2*dim
        P5 = x
        x = S2D_Block(x, dim, is_training, 'S2D_Block4')
        dim = 2*dim
        P6 = x
        x = S2D_Block(x, dim, is_training, 'S2D_Block5')
        # dim = 2*dim
        P7 = x
        return P3, P4, P5, P6, P7


if __name__ == '__main__':
    fake_input = tf.constant(2, shape=[4, 1280, 768, 3])
    P3, P4, P5, P6, P7 = S2DChain(fake_input, 32, "S2DChain")
    print(P3, P4, P5, P6, P7)






