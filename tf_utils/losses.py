import tensorflow as tf

def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, label_smoothing=0, from_logits=False, sparse=False):
    """Implementation of Focal Loss from the paper in multiclass classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)

    """
    y_pred = tf.cast(y_pred, tf.float32)
    if sparse:
        y_true = tf.one_hot(y_true, tf.shape(y_pred)[-1], axis=-1, dtype=tf.float32)
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.float32)

    # clip to prevent NaN's and Inf's
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

    num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
    y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    # Calculate focal loss
    loss = - y_true * (alpha * tf.pow((1 - y_pred), gamma) * tf.math.log(y_pred))
    return loss

def sv_softmax_loss(y_true, logits, t=1.2, s=1, label_smoothing=0, normalize=False, sparse=False):
    #https://github.com/comratvlad/sv_softmax/blob/master/src/custom_losses.py

    logits = tf.cast(logits, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    zeros = tf.zeros_like(logits, dtype=tf.float32)
    ones = tf.ones_like(logits, dtype=tf.float32)

    if sparse:
        y_true = tf.one_hot(y_true, tf.shape(logits)[-1], axis=-1, dtype=tf.float32)
    if normalize:
        logits = tf.math.l2_normalize(logits, axis=-1)
    y_true = tf.cast(y_true, tf.float32)

    logit_y = tf.reduce_sum(tf.multiply(y_true, logits), axis=-1, keepdims=True)
    I_k = tf.where(logit_y >= logits, zeros, ones)

    h = tf.exp(s * tf.multiply(t - 1., tf.multiply(logits + 1., I_k)))

    softmax = tf.exp(s * logits) / (tf.reduce_sum(tf.multiply(tf.exp(s * logits), h), axis=-1, keepdims=True) + epsilon)

    # We add epsilon because log(0) = nan
    softmax = tf.add(softmax, epsilon)

    num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
    y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    loss = tf.multiply(y_true, -tf.math.log(softmax))
    return loss

def arcface_loss(y_true, logits, s=30, m=0.5, easy_margin=False, label_smoothing=0, sparse=False):
    logits = tf.cast(logits, tf.float32)

    if sparse:
        y_true = tf.one_hot(y_true, tf.shape(logits)[-1], axis=-1, dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
    y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    cosine = tf.math.l2_normalize(logits, axis=-1)
    sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))

    cos_m = tf.math.cos(m)
    sin_m = tf.math.sin(m)
    th = tf.math.cos(math.pi - m)
    mm = tf.math.sin(math.pi - m) * m

    phi = cosine * cos_m - sine * sin_m
    if easy_margin:
        phi = tf.where(cosine > 0, phi, cosine)
    else:
        phi = tf.where(cosine > th, phi, cosine - mm)

    output = (y_true * phi) + ((1.0 - y_true) * cosine)
    output *= s
    loss = tf.keras.losses.categorical_crossentropy(y_true, output, from_logits=True)
    return loss

class CategoricalFocalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0, from_logits=True, sparse=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits
        self.sparse = sparse
    def call(self, y_true, y_pred):
        loss = categorical_focal_loss(y_true, y_pred,
                                      gamma=self.gamma,
                                      alpha=self.alpha,
                                      label_smoothing=self.label_smoothing,
                                      from_logits=self.from_logits,
                                      sparse=self.sparse)
        N = tf.cast(tf.shape(y_true)[0], tf.float32)
        return tf.reduce_sum(loss) / N

class SVSoftmaxLoss(tf.keras.losses.Loss):
    def __init__(self, t=1.2, s=1, label_smoothing=0, normalize=False, sparse=True, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.s = s
        self.label_smoothing = label_smoothing
        self.normalize = normalize
        self.sparse = sparse
    def call(self, y_true, logits):
        loss = sv_softmax_loss(y_true, logits,
                               s=self.s,
                               t=self.t,
                               label_smoothing=self.label_smoothing,
                               normalize=self.normalize,
                               sparse=self.sparse)
        N = tf.cast(tf.shape(y_true)[0], tf.float32)
        return tf.reduce_sum(loss) / N

class ArcfaceLoss(tf.keras.losses.Loss):
    def __init__(self, s=30, m=0.3, easy_margin=False, label_smoothing=0, sparse=True, **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.s = s
        self.easy_margin = easy_margin
        self.label_smoothing = label_smoothing
        self.sparse = sparse
    def call(self, y_true, logits):
        loss = arcface_loss(y_true, logits,
                               s=self.s,
                               m=self.m,
                               easy_margin=self.easy_margin,
                               label_smoothing=self.label_smoothing,
                               sparse=self.sparse)
        N = tf.cast(tf.shape(y_true)[0], tf.float32)
        return tf.reduce_sum(loss) / N
