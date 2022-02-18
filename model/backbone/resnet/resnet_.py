from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.contrib.slim.nets import resnet_v1
# from tensorflow.contrib.slim.nets import resnet_utils
# from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import tensorflow as tf
from model.backbone.resnet import resnet_v1
from model.backbone.resnet import resnet_utils
from model.backbone.resnet.resnet_v1 import resnet_v1_block



# from resnet_utils import resnet_arg_scope
# from resnet_v1 import resnet_v1

slim = tf.contrib.slim


def resnet_arg_scope(is_training = True,
                     weight_decay = 0.0004,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-3,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu6,
                     use_batch_norm=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
    
#       """
#       Defines the default ResNet arg scope.

#       """
    batch_norm_params = {
      'is_training': False,#is_training,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'trainable': False,#is_training,
      'updates_collections': batch_norm_updates_collections,
      'fused': None  # Use fused batch norm if possible.
      }
    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03),
        trainable=is_training,
        activation_fn=activation_fn,
        normalizer_fn=slim.batch_norm if use_batch_norm else None,
        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

def resnet_base(in_put, is_training, scope_name):
    blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
    
    
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            net = resnet_utils.conv2d_same(
                in_put, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')
            
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1],
                                                global_pool=False,
                                                output_stride=None,
                                                store_non_strided_activations=True,
                                                include_root_block=False,
                                                scope=scope_name)
        
    with slim.arg_scope(resnet_arg_scope(is_training=(is_training))):
        C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                blocks[1:2],
                                                global_pool=False,
                                                output_stride=None,
                                                store_non_strided_activations=True,
                                                include_root_block=False,
                                                scope=scope_name)
        
    with slim.arg_scope(resnet_arg_scope(is_training=(is_training))):
        C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                                blocks[2:3],
                                                global_pool=False,
                                                output_stride=None,
                                                store_non_strided_activations=True,
                                                include_root_block=False,
                                                scope=scope_name)
        
    with slim.arg_scope(resnet_arg_scope(is_training=(is_training))):
        C5, end_points_C5 = resnet_v1.resnet_v1(C4,
                                                blocks[3:4],
                                                global_pool=False,
                                                output_stride=None,
                                                store_non_strided_activations=True,
                                                include_root_block=False,
                                                scope=scope_name)
    feature_dict = {
        'p3': end_points_C3[scope_name+'/block2/unit_4/bottleneck_v1'],
        'p4': end_points_C4[scope_name+'/block3/unit_6/bottleneck_v1'],
        'p5': end_points_C5[scope_name+'/block4/unit_3/bottleneck_v1']
    }    

    return feature_dict
    
    