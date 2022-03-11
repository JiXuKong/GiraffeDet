import tensorflow as tf

from model.layers import conv2d, bn_, SiLU, upsample_layer, downsample_layer

'''
2022/02/14 23:00.
I'm Not Sure There need any activations or large Kernel conv(like 3x3) or BatchNorm.
2022/02/14 23:31 today job down.
'''



def conv_bn_act(x, dim, k=3, scope=None):
    with tf.variable_scope(scope):
        x = conv2d(x, dim, k, scope=scope + 'conv')
        x = bn_(x, scope=scope + "BN")
        x = SiLU(x)
        return x

class FPN(object):
    '''
    For The First Layer.
    '''
    def __init__(self, dim):
        self.dim = dim
    
    def forward(self, x):
        with tf.variable_scope("FirstLayerTD"):
            P3, P4, P5, P6, P7 = x
            P3 = conv2d(P3, self.dim, 1, scope='ProjectionP3')
            P4 = conv2d(P4, self.dim, 1, scope='ProjectionP4')
            P5 = conv2d(P5, self.dim, 1, scope='ProjectionP5')
            P6 = conv2d(P6, self.dim, 1, scope='ProjectionP6')
            P7 = conv2d(P7, self.dim, 1, scope='ProjectionP7')

            P3_f = conv_bn_act(P3, self.dim, scope='P3P3Cat')

            P4 = tf.concat([downsample_layer(P3, scope='P3DownSample'), P4], axis=-1, name="P3P4Cat")
            P4 = conv2d(P4, self.dim, 1, scope='ProjectionP3P4Cat')
            P4_f = conv_bn_act(P4, self.dim, scope='P3P4Cat')

            P5 = tf.concat([downsample_layer(P4, scope='P4DownSample'), P5], axis=-1, name="P4P5Cat")
            P5 = conv2d(P5, self.dim, 1, scope='ProjectionP4P5Cat')
            P5_f = conv_bn_act(P5, self.dim, scope='P4P5Cat')

            P6 = tf.concat([downsample_layer(P5, scope='P5DownSample'), P6], axis=-1, name="P5P6Cat")
            P6 = conv2d(P6, self.dim, 1, scope='ProjectionP5P6Cat')
            P6_f = conv_bn_act(P6, self.dim, scope='P5P6Cat')

            P7 = tf.concat([downsample_layer(P6, scope='P6DownSample'), P7], axis=-1, name="P6P7Cat")
            P7 = conv2d(P7, self.dim, 1, scope='ProjectionP6P7Cat')
            P7_f = conv_bn_act(P7, self.dim, scope='P6P7Cat')
            return [P3_f, P4_f, P5_f, P6_f, P7_f]


class BUQueen(object):
    def __init__(self, dim, scope):
        self.dim = dim
        self.scope = scope
    
    def forward(self, x):
        with tf.variable_scope("QueenBULayer" + self.scope):
            P3, P4, P5, P6, P7, skip_P7, skip_P3 = x
            P3_d = downsample_layer(P3, scope='P3DownSample')
            P4_d = downsample_layer(P4, scope='P4DownSample')
            P5_d = downsample_layer(P5, scope='P5DownSample')
            P6_d = downsample_layer(P6, scope='P6DownSample')

            P4_u = upsample_layer(P4, tf.shape(P3)[1:3], scope='P4UpSample')
            P5_u = upsample_layer(P5, tf.shape(P4)[1:3], scope='P5UpSample')
            P6_u = upsample_layer(P6, tf.shape(P5)[1:3], scope='P6UpSample')
            P7_u = upsample_layer(P7, tf.shape(P6)[1:3], scope='P7UpSample')
            
            if skip_P3 is not None:
                P3_ = tf.concat([skip_P3, P4_u, P3], axis=-1, name="P3P4_u")
            else:
                P3_ = tf.concat([P4_u, P3], axis=-1, name="P3P4_u")
            P3_ = conv2d(P3_, self.dim, 1, scope='ProjectionP3P4_uCat')
            P3_f = conv_bn_act(P3_, self.dim, scope='P3_f')

            P3_d_ = downsample_layer(P3_, scope='P3_DownSample')
            P4_ = tf.concat([P5_u, P4, P3_d, P3_d_], axis=-1, name="P5_uP4P3_d")
            P4_ = conv2d(P4_, self.dim, 1, scope='ProjectionP5_uP4P3_dCat')
            P4_f = conv_bn_act(P4_, self.dim, scope='P4_f')

            P4_d_ = downsample_layer(P4_, scope='P4_DownSample')
            P5_ = tf.concat([P6_u, P5, P4_d, P4_d_], axis=-1, name="P6_uP5P4_d")
            P5_ = conv2d(P5_, self.dim, 1, scope='ProjectionP6_uP5P4_dCat')
            P5_f = conv_bn_act(P5_, self.dim, scope='P5_f')

            P5_d_ = downsample_layer(P5_, scope='P5_DownSample')
            P6_ = tf.concat([P7_u, P6, P5_d, P5_d_], axis=-1, name="P7_uP6P5_d")
            P6_ = conv2d(P6_, self.dim, 1, scope='ProjectionP7_uP6P5_dCat')
            P6_f = conv_bn_act(P6_, self.dim, scope='P6_f')

            P6_d_ = downsample_layer(P6_, scope='P6_DownSample')
            if skip_P7 is not None:
                P7_ = tf.concat([skip_P7, P7, P6_d, P6_d_], axis=-1, name="P7P6_d")
            else:
                P7_ = tf.concat([P7, P6_d, P6_d_], axis=-1, name="P7P6_d")
            P7_f = conv_bn_act(P7_, self.dim, scope='P7_f')
            
            return [P3_f, P4_f, P5_f, P6_f, P7_f]


class TDQueen(object):
    def __init__(self, dim, scope):
        self.dim = dim
        self.scope = scope
    
    def forward(self, x):
        with tf.variable_scope("QueenTDLayer" + self.scope):
            P3, P4, P5, P6, P7, skip_P7, skip_P3 = x
            P3_d = downsample_layer(P3, scope='P3DownSample')
            P4_d = downsample_layer(P4, scope='P4DownSample')
            P5_d = downsample_layer(P5, scope='P5DownSample')
            P6_d = downsample_layer(P6, scope='P6DownSample')

            P4_u = upsample_layer(P4, tf.shape(P3)[1:3], scope='P4UpSample')
            P5_u = upsample_layer(P5, tf.shape(P4)[1:3], scope='P5UpSample')
            P6_u = upsample_layer(P6, tf.shape(P5)[1:3], scope='P6UpSample')
            P7_u = upsample_layer(P7, tf.shape(P6)[1:3], scope='P7UpSample')

            if skip_P7 is not None:
                P7_ = tf.concat([skip_P7, P7, P6_d], axis=-1, name="P7P6_d")
                P7_ = conv2d(P7_, self.dim, 1, scope='Projectionskip_P7P7P6_dCat')
            else:
                P7_ = tf.concat([P7, P6_d], axis=-1, name="P7P6_d")
                P7_ = conv2d(P7_, self.dim, 1, scope='ProjectionP7P6_dCat')
            P7_f = conv_bn_act(P7_, self.dim, scope='P7_f')

            P7_u_ = upsample_layer(P7_, tf.shape(P6)[1:3], scope='P7_UpSample')
            P6_ = tf.concat([P7_u, P6, P5_d, P7_u_], axis=-1, name="P7_uP6P5_d")
            P6_ = conv2d(P6_, self.dim, 1, scope='ProjectionP7_uP6P5_dCat')
            P6_f = conv_bn_act(P6_, self.dim, scope='P6_f')

            P6_u_ = upsample_layer(P6_, tf.shape(P5)[1:3], scope='P6_UpSample')
            P5_ = tf.concat([P6_u, P5, P4_d, P6_u_], axis=-1, name="P6_uP5P4_d")
            P5_ = conv2d(P5_, self.dim, 1, scope='ProjectionP6_uP5P4_dCat')
            P5_f = conv_bn_act(P5_, self.dim, scope='P5_f')

            P5_u_ = upsample_layer(P5_, tf.shape(P4)[1:3], scope='P5_UpSample')
            P4_ = tf.concat([P5_u, P4, P3_d, P5_u_], axis=-1, name="P5_uP4P3_d")
            P4_ = conv2d(P4_, self.dim, 1, scope='ProjectionP5_uP4P3_dCat')
            P4_f = conv_bn_act(P4_, self.dim, scope='P4_f')

            P4_u_ = upsample_layer(P4_, tf.shape(P3)[1:3], scope='P4_UpSample')
            if skip_P3 is not None:
                P3_ = tf.concat([skip_P3, P4_u, P3, P4_u_], axis=-1, name="P3P4_u")
            else:
                P3_ = tf.concat([P4_u, P3, P4_u_], axis=-1, name="P3P4_u")
            P3_ = conv2d(P3_, self.dim, 1, scope='ProjectionP4_uP3Cat')
            P3_f = conv_bn_act(P3_, self.dim, scope='P3_f')
            return [P3_f, P4_f, P5_f, P6_f, P7_f]


class GFPN(object):
    '''
    first layers' top-down structure
    top-down Queen-fusion
    bottom-up Queen-fusion
    consider skip connection(with an outside cat func)
    '''
    def __init__(self, dim, fai_d, fai_w, depth):
        self.dim = dim
        self.fai_d = fai_d
        self.fai_w = fai_w
        self.depth = depth
        self.width = int(self.dim * self.fai_w)
        self.f_dict = {}
    
    def forward(self, x):
        self.f_dict["x_0"] = FPN(self.width).forward(x)
        self.f_dict["x_1"] = TDQueen(self.width, scope="1").forward(self.f_dict["x_0"] + [None, None]) 
        for d in range(2, self.depth):
            if d%2 == 0: 
                skip_f_7 = []
                skip_f_3 = []
                for i in range(0, d, 2):
                    key = "x_" + str(i)
                    skip_f_7.append(self.f_dict[key][4])#get P7
                    skip_f_3.append(self.f_dict[key][0])#get P3
                skip_f_7 = tf.concat(skip_f_7, axis=-1)
                skip_f_3 = tf.concat(skip_f_3, axis=-1)
                self.f_dict["x_" + str(d)] = BUQueen(self.width, scope=str(d)).forward(self.f_dict["x_" + str(d-1)] + [skip_f_7] + [skip_f_3])
            else:
                skip_f_7 = []
                skip_f_3 = []
                for i in range(1, d, 2):
                    key = "x_" + str(i)
                    skip_f_7.append(self.f_dict[key][4])#get P7
                    skip_f_3.append(self.f_dict[key][0])#get P3
                skip_f_7 = tf.concat(skip_f_7, axis=-1)
                skip_f_3 = tf.concat(skip_f_3, axis=-1)
                self.f_dict["x_" + str(d)] = TDQueen(self.width, scope=str(d)).forward(self.f_dict["x_" + str(d-1)] + [skip_f_7] + [skip_f_3])
        
        return self.f_dict["x_" + str(d)]


