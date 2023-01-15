import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, strides=1, dilation_rate=1, use_bias=False, padding='same', activation='relu', conv_type='normal',
                        norm_type='batchnorm', batchnorm_momentum=0.9, layernorm_epsilon=1e-5, kernel_initializer='he_uniform', bias_initializer='zeros', **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.channels=channels
        self.kernel_size=kernel_size
        self.dilation_rate=dilation_rate
        self.use_bias=use_bias
        self.strides=strides
        self.padding=padding
        self.activation=activation
        self.batchnorm_momentum=batchnorm_momentum
        self.layernorm_epsilon=layernorm_epsilon
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.conv_type = conv_type
        if conv_type == 'normal':
            self.conv = tf.keras.layers.Conv2D(channels, kernel_size, strides, dilation_rate=dilation_rate, padding=padding,
                                                use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        elif conv_type == 'separable':
            self.conv = tf.keras.layers.SeparableConv2D(channels, kernel_size, strides, dilation_rate=dilation_rate, padding=padding,
                                                use_bias=use_bias, depthwise_initializer=kernel_initializer, pointwise_initializer=kernel_initializer, bias_initializer=bias_initializer)
        elif conv_type == 'depthwise':
            self.conv = tf.keras.layers.DepthwiseConv2D(kernel_size, strides, dilation_rate=dilation_rate, padding=padding,
                                                use_bias=use_bias, depthwise_initializer=kernel_initializer, bias_initializer=bias_initializer)
        else: raise ValueError
        self.norm_type = norm_type
        if norm_type == 'batchnorm':
            self.norm = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)
        elif norm_type == 'layernorm':
            self.norm = tf.keras.layers.LayerNormalization(epsilon=layernorm_epsilon)
        elif norm_type is None:
            self.norm = None
        else: raise ValueError
        self.act = tf.keras.layers.Activation(activation)
    
    def call(self, input):
        x = self.conv(input)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels':self.channels,
            'kernel_size':self.kernel_size,
            'strides':self.strides,
            'dilation_rate':self.dilation_rate,
            'use_bias':self.use_bias,
            'padding':self.padding,
            'activation':self.activation,
            'conv_type':self.conv_type,
            'norm_type':self.norm_type,
            'batchnorm_momentum':self.batchnorm_momentum,
            'layerorm_epsilon':self.layernorm_epsilon,
            'kernel_initializer':self.kernel_initializer,
            'bias_initializer':self.bias_initializer,
        })
        return config
      
