import tensorflow as tf

EPS = 1e-12

# leaky relu function
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def get_conv_weight(name,kshape,wd=0.0005):
    #with tf.device('/cpu:0'):
    var=tf.get_variable(name,shape=kshape,initializer=tf.contrib.layers.xavier_initializer())
    if wd!=0:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

def conv3d(name,l_input,kernel_size, strides=[1,1,1,1,1], padding="SAME"):
    conv_w=get_conv_weight(name=name+'_conv3d', kshape=kernel_size)
    bias_w=get_conv_weight(name+'_bias',[kernel_size[-1]],0)
    
    conv = tf.nn.bias_add(tf.nn.conv3d(l_input, conv_w, strides=strides, padding=padding), bias_w, name=name)
    return conv
    
def convS(name,l_input,in_channels,out_channels, strides=[1,1,1,1,1], padding='SAME'):
    conv_w=get_conv_weight(name=name+'_conv3d', kshape=[1,3,3,in_channels,out_channels])
    bias_w=get_conv_weight(name+'_bias', [out_channels], 0)
    
    conv = tf.nn.bias_add(tf.nn.conv3d(l_input, conv_w, strides=strides, padding=padding), bias_w, name=name)
    return conv
    
def convT(name,l_input,in_channels,out_channels, n_t=3, strides=[1,1,1,1,1], padding='SAME'):
    conv_w=get_conv_weight(name=name+'_conv3d', kshape=[n_t,1,1,in_channels,out_channels])
    bias_w=get_conv_weight(name+'_bias', [out_channels], 0)
    
    conv = tf.nn.bias_add(tf.nn.conv3d(l_input, conv_w, strides=strides, padding=padding), bias_w, name=name)
    return conv
    
def deconv3D(name, l_input, out_shape, kernel_size, strides=[1,1,1,1,1], padding='SAME'):
    # output shape : (b, temporal depth, w, h, out_ch)
    # filters : output_chnnel size
    # kernel_size : size of kernel [kernel_temporal, filter_w, filter_h, out_ch, in_ch]
    # strides : size of kernel [temporal, spatio_w, spatio_h, out_ch, in_ch]
    # tf.nn.conv3d_transpose(input, filter=get_conv_weight
    deconv_w=get_conv_weight(name=name+'_deconv3d', kshape=kernel_size)
    deconv = tf.nn.conv3d_transpose(l_input, filter=deconv_w, output_shape=out_shape,
                                    strides=strides, padding=padding, name=name+'_deconv')
    return deconv
    
def deconv3D2(name, l_input, out_shape, kernel_size=[3,3,3], strides=(2,2,2), padding='SAME'):
    deconv=tf.layers.conv3d_transpose(l_input, filters=out_shape[-1], kernel_size=kernel_size, strides=strides, padding=padding, name=name+'_deconv3d')

    return deconv

def conv3d_transpose(name, l_input, w, b, output_shape, stride=1):
    transp_conv = tf.nn.conv3d_transpose(l_input, w, output_shape, strides=[1, stride, stride, stride, 1], padding='SAME')
    return tf.nn.bias_add(transp_conv, b, name=name)