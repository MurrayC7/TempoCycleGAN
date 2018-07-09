import tensorflow as tf
from TensorFlow import ops


class Discriminator:
    def __init__(self, name, is_training, norm='instance', use_sigmoid=False, is_tempo):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.use_sigmoid = use_sigmoid
        self.is_tempo = is_tempo

    def __call__(self, input):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """

        # add the temporal discriminator 6/22
        if not self.is_tempo:
            with tf.variable_scope(self.name):
                # convolution layers
                C32 = ops.Ck(input, 32, reuse=self.reuse, norm=None,
                             is_training=self.is_training, name='C32')  # (?, w, h, 32)
                C64 = ops.Ck(C32, 64, reuse=self.reuse, norm=self.norm,
                             is_training=self.is_training, name='C64')  # (?, w/2, h/2, 64)
                C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                              is_training=self.is_training, name='C128')  # (?, w/4, h/4, 128)
                C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                              is_training=self.is_training, name='C256')  # (?, w/8, h/8, 256)

                # apply a convolution to produce a 1 dimensional output (1 channel?)
                # use_sigmoid = False if use_lsgan = True
                output = ops.last_conv(C256, reuse=self.reuse,
                                       use_sigmoid=self.use_sigmoid, name='output')  # (?, w/16, h/16, 1)

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output
        else:
            with tf.variable_scope(self.name):
                # convolution layers
                advected_input = self.advect(input)
                C32 = ops.Ck(advected_input, 32, reuse=self.reuse, norm=None,
                             is_training=self.is_training, name='C32')  # (?, w, h, 32)
                C64 = ops.Ck(C32, 64, reuse=self.reuse, norm=self.norm,
                             is_training=self.is_training, name='C64')  # (?, w/2, h/2, 64)
                C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                              is_training=self.is_training, name='C128')  # (?, w/4, h/4, 128)
                C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                              is_training=self.is_training, name='C256')  # (?, w/8, h/8, 256)

                # apply a convolution to produce a 1 dimensional output (1 channel?)
                # use_sigmoid = False if use_lsgan = True
                output = ops.last_conv(C256, reuse=self.reuse,
                                       use_sigmoid=self.use_sigmoid, name='output')  # (?, w/16, h/16, 1)

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output

    def advect(self, input):
        """
        The advection layer to process the sequence {x_t-1, x_t, x_t+1}
        :param input:
        :return:
        """
        return 0
