import tensorflow as tf
from config import *
from tf_utils import *

class P3DBottleNeck():
    def __init__(self, l_input, in_channels, channels, out_channels, n_t=3, downsample=False, _id=0, pType="A", activation=True):
        self.X = l_input
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels
        self.n_t = n_t
        self.pType = pType
        self.id = _id
        self.downsample = downsample
        if downsample:
            self.strides=[1,1,2,2,1]
        else:
            self.strides=[1,1,1,1,1]
        self.activation=activation
    
    def P3D_A(self, name, x):
        x=convS(name+'_S', x, self.channels, self.channels)
        x=tf.layers.batch_normalization(x, training=IS_TRAIN)
        x=tf.nn.relu(x)
        
        x=convT(name+'_T', x, self.channels, self.channels, n_t=self.n_t)
        x=tf.layers.batch_normalization(x, training=IS_TRAIN)
        x=tf.nn.relu(x)
        return x
    
    def P3D_B(self, name, x):
        x_s=convS(name+'_S', x, self.channels, self.channels)
        x_s=tf.layers.batch_normalization(x_s, training=IS_TRAIN)
        x_s=tf.nn.relu(x_s)
        
        x_t=convT(name+'_T', x, self.channels, self.channels, n_t=self.n_t)
        x_t=tf.layers.batch_normalization(x_t, training=IS_TRAIN)
        x_t=tf.nn.relu(x_s)
        return x_s + x_t
    
    def P3D_C(self, name, x):
        x_s=convS(name+'_S', x, self.channels, self.channels)
        x_s=tf.layers.batch_normalization(x_s, training=IS_TRAIN)
        x_s=tf.nn.relu(x_s)
        
        x_st=convT(name+'_T', x_s, self.channels, self.channels, n_t=self.n_t)
        x_st=tf.layers.batch_normalization(x_st, training=IS_TRAIN)
        x_st=tf.nn.relu(x_st)
        return x_s + x_st
    
    def build(self):
        residual = self.X
        
        # 1x1x1 conv : in_channels --> channels
        conv_w=get_conv_weight('conv_1w_{}'.format(self.id), [1, 1, 1, self.in_channels, self.channels])
        bias_w=get_conv_weight('conv_1b_{}'.format(self.id), [self.channels])
        out=tf.nn.conv3d(self.X, conv_w, strides=self.strides, padding='SAME')
        out=tf.nn.bias_add(out, bias_w, name='conv_1_{}'.format(self.id))
        out=tf.layers.batch_normalization(out, training=IS_TRAIN)
        out=tf.nn.relu(out)
        
        # P3D : channels --> channels
        if self.pType == "A":
            out=self.P3D_A(name="P3D_A_{}".format(self.id), x=out)
        elif self.pType == "B":
            out=self.P3D_B(name="P3D_B_{}".format(self.id), x=out)
        else:
            out=self.P3D_C(name="P3D_C_{}".format(self.id), x=out)
                
        # Residual
        # 1x1x1 conv : channels --> out_channels
        conv_w=get_conv_weight('conv_2w_{}'.format(self.id), [1, 1, 1, self.channels, self.out_channels])
        bias_w=get_conv_weight('conv_2b_{}'.format(self.id), [self.out_channels])
        out=tf.nn.conv3d(out, conv_w, strides=[1,1,1,1,1], padding='SAME')
        out=tf.nn.bias_add(out, bias_w, name='conv_2_{}'.format(self.id))
        out=tf.layers.batch_normalization(out, training=IS_TRAIN)
        #out=tf.nn.relu(out)
        
        # down-sampling residual : in_channels --> out_chaneels
        conv_w=get_conv_weight('dw3d_w_{}'.format(self.id), [1, 1, 1, self.in_channels, self.out_channels])
        bias_w=get_conv_weight('dw3d_b_{}'.format(self.id), [self.out_channels])
        residual=tf.nn.conv3d(residual, conv_w, strides=self.strides, padding='SAME')
        residual=tf.nn.bias_add(residual, bias_w, name='dw3d_{}'.format(self.id))
        residual=tf.layers.batch_normalization(residual, training=IS_TRAIN)
        
        if self.activation:    
            out += residual
            out = tf.nn.relu(out, name="{}_btn_P3D_{}".format(self.id, self.pType))
        else:
            out = tf.add(out, residual, name="{}_btn_P3D_{}".format(self.id, self.pType))
        return out


class buildP3DBlock():
    def __init__(self, l_input, in_channels, channels, out_channels, iteration, cnt,
                 n_t=3, downsample=False, last_activation=True):
        self.input = l_input
        self.in_ch = in_channels
        self.ch = channels
        self.out_ch = out_channels
        self.iteration = iteration
        self.cnt = cnt
        self.n_t = n_t
        self.ptype = ["A", "B", "C"]
        self.last_activation = last_activation
        self.downsample=downsample
    
    def build(self):
        len_ptype=len(self.ptype)
        
        ptype=self.ptype[self.cnt%len_ptype]
        x = P3DBottleNeck(l_input=self.input,
                          in_channels=self.in_ch, 
                          channels=self.ch, 
                          out_channels=self.out_ch,
                          n_t=self.n_t,
                          _id=self.cnt, 
                          downsample=self.downsample,
                          pType=ptype).build()
        #print(x)
        
        last_iter = self.cnt + self.iteration - 1
        for i in range(self.cnt + 1, self.cnt + self.iteration):
            ptype=self.ptype[i%len_ptype]
            if i < last_iter:
                x = P3DBottleNeck(l_input=x, 
                                  in_channels=self.out_ch, 
                                  channels=self.ch, 
                                  out_channels=self.out_ch, 
                                  n_t=self.n_t,
                                  _id=i, 
                                  pType=ptype).build()
            else:
                x = P3DBottleNeck(l_input=x, 
                                  in_channels=self.out_ch, 
                                  channels=self.ch, 
                                  out_channels=self.out_ch, 
                                  n_t=self.n_t,
                                  _id=i, 
                                  pType=ptype,
                                  activation=self.last_activation).build()
        return x
