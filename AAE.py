import tensorflow as tf
from config import *
from tf_utils import *
from P3D import buildP3DBlock


class AAE():
    def __init__(self, img_ch=1, n_t=2, latent_dim=LATENT_DIM): 
        '''
          Parameters
             x   : input tensor
             n_t : step size of temporal direction (used in P3D, convT())
        '''
        #self.x = x
        self.n_t = n_t
        self.img_ch = img_ch
        self.z_dim = latent_dim
        #self.input_shape = self.x.get_shape().as_list()
    
    def encoder(self, _x):
        w_list=[]
        b_list=[]
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):    
            cnt=0
            # _x: [batch, t, h, w, 3]
            # x: [batch, t, h, w, 32]
            conv_w=get_conv_weight('AAE_E_conv1_w', [1, 3, 3, self.img_ch, 32])
            bias_w=get_conv_weight('AAE_E_conv1_b', [32])
            x=tf.nn.conv3d(_x, conv_w, strides=[1,1,1,1,1], padding='SAME')
            x=tf.nn.bias_add(x, bias_w, name='AAE_E_conv1')
            x=tf.layers.batch_normalization(x, training=IS_TRAIN)
            x=tf.nn.relu(x)
            
            w_list.append(conv_w)
            b_list.append(bias_w)
            #print(x)

            # x: [batch, t, h, w, 32]
            iteration=3
            x=buildP3DBlock(x, 32, 16, 32, iteration=iteration, cnt=cnt, n_t=self.n_t).build()
            cnt += iteration
            
            # x: [batch, t/2, h/2, w/2, 32]
            x= tf.nn.max_pool3d(x, [1, 3, 3, 3, 1], strides=[1, self.n_t, 2, 2, 1], padding='SAME')

            # x: [batch, t/2, h/2, w/2, 64]
            iteration=5
            x=buildP3DBlock(x, 32, 16, 64, iteration=iteration, cnt=cnt, n_t=self.n_t).build()
            cnt += iteration

            # x: [batch, t/4, h/4, w/4, 64]
            x= tf.nn.max_pool3d(x, [1, 3, 1, 1, 1], strides=[1, self.n_t, 2, 2, 1], padding='SAME')

            # x: [batch, t/4, h/4, w/4, 64]
            iteration=5
            x=buildP3DBlock(x, 64, 32, 64, iteration=iteration, cnt=cnt, n_t=self.n_t).build()
            cnt += iteration

            # x: [batch, t/8, h/4, w/4, 64]
            x= tf.nn.max_pool3d(x, [1, 3, 3, 3, 1], strides=[1, self.n_t, 1, 1, 1], padding='SAME')

            # x: [batch, t/8, h/4, w/4, 128]
            iteration=8
            x=buildP3DBlock(x, 64, 32, 128, iteration=iteration, cnt=cnt, n_t=self.n_t, 
                         last_activation=False).build()
            cnt += iteration
            
            conv_w=get_conv_weight('AAE_E_conv2_w', [1, 3, 3, 128, self.z_dim])
            bias_w=get_conv_weight('AAE_E_conv2_b', [self.z_dim])
            x=tf.nn.conv3d(x, conv_w, strides=[1,1,1,1,1], padding='SAME')
            x=tf.nn.bias_add(x, bias_w, name='AAE_E_conv2')

            w_list.append(conv_w)
            b_list.append(bias_w)    
        return x
    
    def decoder(self, _x):
        w_list=[]
        b_list=[]
        
        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
            cnt=21
            # _x: [batch, t/8, h/4, w/4, 128]
            iteration=8
            x=buildP3DBlock(_x, self.z_dim, 32, 64, iteration=iteration, cnt=cnt, n_t=self.n_t).build()
            cnt += iteration

            # x: [batch, t/4, h/4, w/4, 64]
            x=tf.layers.conv3d_transpose(x, x.get_shape().as_list()[4], kernel_size=[3,3,3], strides=(2,1,1), padding="SAME", name="AAE_D_deconv1_deconv3d")
            
            # x: [batch, t/4, h/4, w/4, 64]
            iteration=5
            x=buildP3DBlock(x, 64, 32, 64, iteration=iteration, cnt=cnt, n_t=self.n_t).build()
            cnt += iteration

            # x: [batch, t/2, h/2, w/2, 64]
            x=tf.layers.conv3d_transpose(x, x.get_shape().as_list()[4], kernel_size=[3,3,3], strides=(2,2,2), padding="SAME", name="AAE_D_deconv2_deconv3d")
            
            # x: [batch, t/2, h/2, w/2, 32]
            iteration=5
            x=buildP3DBlock(x, 64, 16, 32, iteration=iteration, cnt=cnt, n_t=self.n_t).build()
            cnt += iteration

            # x: [batch, t, h, w, 32]
            x=tf.layers.conv3d_transpose(x, x.get_shape().as_list()[4], kernel_size=[3,3,3], strides=(2,2,2), padding="SAME", name="AAE_D_deconv3_deconv3d")
            
            iteration=3
            x=buildP3DBlock(x, 32, 16, 32, iteration=iteration, cnt=cnt, n_t=self.n_t).build()
            cnt += iteration

            # x: [batch, t, h, w, 3]
            conv_w=get_conv_weight('AAE_D_conv_w', [1, 3, 3, 32, self.img_ch])
            bias_w=get_conv_weight('AAE_D_conv_b', [self.img_ch])
            x=tf.nn.conv3d(x, conv_w, strides=[1,1,1,1,1], padding='SAME')
            x=tf.nn.bias_add(x, bias_w, name='AAE_D_conv')
            x=tf.layers.batch_normalization(x, training=IS_TRAIN)
            x=tf.nn.sigmoid(x, name='AAE_D_x_hat')

            w_list.append(conv_w)
            b_list.append(bias_w)
            
        return x
    
    def discriminator(self, z):
        w_list=[]
        b_list=[]
        
        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
            # z : [batchsize, t, h, w, c]   ... [none, 1, 40, 40, 128]
            conv_w=get_conv_weight('AAE_Disc_conv1_w', [1, 3, 3, self.z_dim, 64])
            bias_w=get_conv_weight('AAE_Disc_conv1_b', [64])
            h1 =tf.nn.conv3d(z, conv_w, strides=[1,1,2,2,1], padding='SAME')
            h1 =tf.nn.bias_add(h1, bias_w, name='AAE_Disc_conv1')
            h1 =tf.layers.batch_normalization(h1, training=IS_TRAIN)
            h1 = tf.nn.relu(h1)
            
            w_list.append(conv_w)
            b_list.append(bias_w)
            
            # [None, 1, 20, 20, 32]
            conv_w=get_conv_weight('AAE_Disc_conv2_w', [1, 1, 1, 64, 32])
            bias_w=get_conv_weight('AAE_Disc_conv2_b', [32])
            h2 =tf.nn.conv3d(h1, conv_w, strides=[1,1,1,1,1], padding='SAME')
            h2 =tf.nn.bias_add(h2, bias_w, name='AAE_Disc_conv2')
            h2 =tf.layers.batch_normalization(h2, training=IS_TRAIN)
            h2 = tf.nn.relu(h2)
            
            w_list.append(conv_w)
            b_list.append(bias_w)
            
            # [None, 1, 20, 20, 1]
            logits = conv3d("AAE_Disc_logits", h2, [1, 1, 1, 32, 1], strides=[1, 1, 1, 1, 1])
        return tf.nn.sigmoid(logits), logits
    
    def adversarial_autoencoder(self, _x, z_sample):
        #with tf.variable_scope("AAE", reuse=tf.AUTO_REUSE):
        w_list=[]
        b_list=[]

        # encoding
        z = self.encoder(_x)
        #print(z)
        
        # decoding
        y = self.decoder(z)
        
        # Reconstruction Loss
        marginal_likelihood = -tf.reduce_mean(tf.reduce_mean(tf.squared_difference(_x, y)))

        # GAN Loss
        z_real = z_sample
        z_fake = z

        D_real, D_real_logits = self.discriminator(z_real)
        D_fake, D_fake_logits = self.discriminator(z_fake)

        # Discriminator Loss
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
        D_loss = D_loss_real + D_loss_fake

        # Generator Loss
        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        D_loss = tf.reduce_mean(D_loss)
        G_loss = tf.reduce_mean(G_loss)

        return y, z, -marginal_likelihood, D_loss, G_loss
        
    def autoencoder_tmp(self, _x):
        enc = self.encoder(_x)
        
        mu, log_std_sq = tf.split(enc, 2, -1)
        
        eps = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        #z = tf.add(mu, tf.matmul(tf.sqrt(tf.exp(log_std_sq)), eps))
        z = tf.add(mu, tf.sqrt(tf.exp(log_std_sq)) * eps)
        #print("Enc: ", enc)
        #print("z. : ", z)
        x_hat = self.decoder(z)
        x_hat = tf.clip_by_value(x_hat, 1e-8, 1 - 1e-8)

        # Loss
        marginal_likelihood = tf.reduce_sum(_x * tf.log(x_hat) + (1 - _x) * tf.log(1 - x_hat), -1)
        marginal_likelihood = tf.reduce_mean(marginal_likelihood)

        KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

        
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(1e-8 + log_std_sq) - log_std_sq - 1, -1)
        KL_divergence = tf.reduce_mean(KL_divergence)
        
        ELBO = marginal_likelihood - KL_divergence
        
        loss = -ELBO
        return x_hat, z, enc, loss, -marginal_likelihood, KL_divergence
        
        
    def generate_sample(self, _x):
        z=self.encoder(_x)
        return self.decoder(z)
    
    def build(self, _x):
        1