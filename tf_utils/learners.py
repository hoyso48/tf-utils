import tensorflow as tf

class FGM(tf.keras.Model):
    def __init__(self, *args, delta=0.2, eps=1e-4, start_step=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.eps = eps
        self.start_step = start_step
        
    def train_step_fgm(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        embedding = self.trainable_variables[0]
        embedding_gradients = tape.gradient(loss, [self.trainable_variables[0]])[0]
        embedding_gradients = tf.zeros_like (embedding) + embedding_gradients
        delta = tf.math.divide_no_nan(self.delta * embedding_gradients , tf.math.sqrt(tf.reduce_sum(embedding_gradients**2)) + self.eps)
        self.trainable_variables[0].assign_add(delta)
        with tf.GradientTape() as tape2:
            y_pred = self(x, training=True)
            new_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            if hasattr(self.optimizer, 'get_scaled_loss'):
                new_loss = self.optimizer.get_scaled_loss(new_loss)
        gradients = tape2.gradient(new_loss, self.trainable_variables)
        if hasattr(self.optimizer, 'get_unscaled_gradients'):
            gradients =  self.optimizer.get_unscaled_gradients(gradients)
        self.trainable_variables[0].assign_sub(delta)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # self_loss.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return tf.cond(self._train_counter < self.start_step, lambda:super(FGM, self).train_step(data), lambda:self.train_step_fgm(data))

class AWP(tf.keras.Model):
    def __init__(self, *args, delta=0.1, eps=1e-4, start_step=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.eps = eps
        self.start_step = start_step
        
    def train_step_awp(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        params = self.trainable_variables
        params_gradients = tape.gradient(loss, self.trainable_variables)
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            delta = tf.math.divide_no_nan(self.delta * grad , tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps)
            self.trainable_variables[i].assign_add(delta)
        with tf.GradientTape() as tape2:
            y_pred = self(x, training=True)
            new_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            if hasattr(self.optimizer, 'get_scaled_loss'):
                new_loss = self.optimizer.get_scaled_loss(new_loss)
            
        gradients = tape2.gradient(new_loss, self.trainable_variables)
        if hasattr(self.optimizer, 'get_unscaled_gradients'):
            gradients =  self.optimizer.get_unscaled_gradients(gradients)
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            delta = tf.math.divide_no_nan(self.delta * grad , tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps)
            self.trainable_variables[i].assign_sub(delta)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # self_loss.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return tf.cond(self._train_counter < self.start_step, lambda:super(AWP, self).train_step(data), lambda:self.train_step_awp(data))
