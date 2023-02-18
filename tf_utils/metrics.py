import tensorflow as tf

# class MultilabelMetrics(tf.keras.metrics.Metric):
#     def __init__(self, metric='fbeta', threshold=0.5, beta=1.0, average='micro', num_classes=None, from_logits=False, name=None, **kwargs):
#         if name is None:
#             name = metric
#         if metric not in ['accuracy', 'precision', 'recall', 'fbeta', 'IoU']:
#             raise ValueError
#         if average not in ['micro', 'macro']:
#             raise ValueError
#         super().__init__(name=name, **kwargs)
#         self.metric = metric
#         self.threshold = threshold
#         self.beta = beta
#         self.average = average
#         self.from_logits = from_logits
#         self.num_classes = num_classes
#         self.shape = (num_classes,) if average=='macro' else ()
#         self.tp = self.add_weight(shape=self.shape, dtype=tf.float32, name='tp', initializer='zeros')
#         self.tn = self.add_weight(shape=self.shape, dtype=tf.float32, name='tn', initializer='zeros')
#         self.fn = self.add_weight(shape=self.shape, dtype=tf.float32, name='fn', initializer='zeros')
#         self.fp = self.add_weight(shape=self.shape, dtype=tf.float32, name='fp', initializer='zeros')

#     def decode_prediction(self, y_true, y_pred):
#         y_pred = tf.cast(y_pred, tf.float32)
#         if self.from_logits:
#             y_pred = tf.nn.sigmoid(y_pred)
#         y_true = tf.cast(y_true, tf.float32)

#         y_t = y_true>0.5 #0.5 for possible pre-label smoothing
#         y_p = y_pred>self.threshold

#         return y_t, y_p

#     def update_state(self, y_true, y_pred, sample_weight=None):

#         y_t, y_p = self.decode_prediction(y_true, y_pred)

#         tps = tf.cast(tf.logical_and(y_t, y_p), tf.float32)
#         tns = tf.cast(tf.logical_and(tf.logical_not(y_t), tf.logical_not(y_p)), tf.float32)
#         fns = tf.cast(tf.logical_and(y_t, tf.logical_not(y_p)), tf.float32)
#         fps = tf.cast(tf.logical_and(tf.logical_not(y_t), y_p), tf.float32)
#         if self.average=='macro':
#             # we assume last axis always represents classes
#             tps = tf.reduce_sum(tf.reshape(tps, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(tps, self.num_classes, -1)))
#             tns = tf.reduce_sum(tf.reshape(tns, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(tns, self.num_classes, -1)))
#             fns = tf.reduce_sum(tf.reshape(fns, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(fns, self.num_classes, -1)))
#             fps = tf.reduce_sum(tf.reshape(fps, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(fps, self.num_classes, -1)))
#         else:
#             tps = tf.reduce_sum(tps)
#             tns = tf.reduce_sum(tns)
#             fns = tf.reduce_sum(fns)
#             fps = tf.reduce_sum(fps)

#         self.tp.assign_add(tps)
#         self.tn.assign_add(tns)
#         self.fn.assign_add(fns)
#         self.fp.assign_add(fps)

#     def result(self):
#         if self.metric.lower()=='accuracy':
#             return tf.reduce_mean(tf.math.divide_no_nan(self.tp+self.tn, self.tp+self.fp+self.fn+self.tn))
#         elif self.metric.lower()=='precision':
#             return tf.reduce_mean(tf.math.divide_no_nan(self.tp, self.tp+self.fp))
#         elif self.metric.lower()=='recall':
#             return tf.reduce_mean(tf.math.divide_no_nan(self.tp, self.tp+self.fn))
#         elif self.metric.lower()=='fbeta':
#             return tf.reduce_mean(tf.math.divide_no_nan((self.beta**2+1)*self.tp, (1+self.beta**2)*self.tp+self.beta**2*self.fn+self.fp))
#         elif self.metric.lower()=='iou':
#             return tf.reduce_mean(tf.math.divide_no_nan(self.tp , self.tp + self.fp + self.fn))
#         else: raise NotImplementedError
        
#     def reset_state(self):
#         self.tp.assign(tf.zeros(self.shape))
#         self.tn.assign(tf.zeros(self.shape))
#         self.fn.assign(tf.zeros(self.shape))
#         self.fp.assign(tf.zeros(self.shape))
        
# class CummulativeMultilabelMetrics(tf.keras.metrics.Metric):
#     #cummulative multilabel metrics
#     def __init__(self, name='mF', num_thresholds=20, beta=1.0, average='macro', from_logits=False, **kwargs):
#         if name not in ['AUC', 'mP', 'mR', 'mF', 'mIoU', 'best_f', 'best_th']:
#             raise ValueError
#         if average not in ['micro', 'macro']:
#             raise ValueError
#         super().__init__(name=name, **kwargs)
#         self.name_ = name
#         self.num_thresholds = num_thresholds
#         self.beta = beta
#         self.average = average
#         self.from_logits = from_logits
#         self.tp = self.add_weight(shape=(num_thresholds-1,), dtype=tf.float32, name='tp', initializer='zeros')
#         self.tn = self.add_weight(shape=(num_thresholds-1,), dtype=tf.float32, name='tn', initializer='zeros')
#         self.fn = self.add_weight(shape=(num_thresholds-1,), dtype=tf.float32, name='fn', initializer='zeros')
#         self.fp = self.add_weight(shape=(num_thresholds-1,), dtype=tf.float32, name='fp', initializer='zeros')
        
#     def decode_prediction(self, y_true, y_pred):
#         y_pred = tf.cast(y_pred, tf.float32)
#         if self.from_logits:
#             y_pred = tf.nn.sigmoid(y_pred)
#         y_true = tf.cast(y_true, tf.float32)
#         return y_true, y_pred

#     def update_state(self, y_true, y_pred, sample_weight=None):
      
#         y_true, y_pred = self.decode_prediction(y_true, y_pred)
        
#         arr = tf.TensorArray(tf.float32, (self.num_thresholds-1), clear_after_read=True)
#         for i in tf.range(self.num_thresholds-1):
#             th = 1./tf.cast(self.num_thresholds, tf.float32)*tf.cast(i+1, tf.float32)
#             y_t = y_true>0.5 #0.5 for possible pre-label smoothing
#             y_p = y_pred>th
#             tps = tf.cast(tf.logical_and(y_t, y_p), tf.float32)
#             tns = tf.cast(tf.logical_and(tf.logical_not(y_t), tf.logical_not(y_p)), tf.float32)
#             fns = tf.cast(tf.logical_and(y_t, tf.logical_not(y_p)), tf.float32)
#             fps = tf.cast(tf.logical_and(tf.logical_not(y_t), y_p), tf.float32)
#             if self.average=='macro':
#                 # we assume last axis always represents classes
#                 tps = tf.reduce_mean(tps, -1)
#                 tns = tf.reduce_mean(tns, -1)
#                 fns = tf.reduce_mean(fns, -1)
#                 fps = tf.reduce_mean(fps, -1)
#             tps = tf.reduce_sum(tps)
#             tns = tf.reduce_sum(tns)
#             fns = tf.reduce_sum(fns)
#             fps = tf.reduce_sum(fps)
#             arr = arr.write(i, [tps, tns, fns, fps]) #(n,3)
#         tps, tns, fns, fps= tf.unstack(arr.stack(), axis=1)
#         self.tp.assign_add(tps)
#         self.tn.assign_add(tns)
#         self.fn.assign_add(fns)
#         self.fp.assign_add(fps)

#     def result(self):
#         if self.name_=='mP': #average precision
#             return tf.reduce_mean(tf.math.divide_no_nan(self.tp, self.tp+self.fp))
#         elif self.name_=='mR': #average recall
#             return tf.reduce_mean(tf.math.divide_no_nan(self.tp, self.tp+self.fn))
#         elif self.name_=='mF': #average fbeta
#             return tf.reduce_mean(tf.math.divide_no_nan((self.beta**2+1)*self.tp, (1+self.beta**2)*self.tp+self.beta**2*self.fn+self.fp))
#         elif self.name_=='mIoU':
#             return tf.reduce_mean(tf.math.divide_no_nan(self.tp , self.tp + self.fp + self.fn))
#         elif self.name_=='best_f': #best fbeta
#             afs = tf.math.divide_no_nan((self.beta**2+1)*self.tp, (1+self.beta**2)*self.tp+self.beta**2*self.fn+self.fp)
#             return tf.reduce_max(afs)
#         elif self.name_=='best_th': #threshold of best fbeta
#             afs = tf.math.divide_no_nan((self.beta**2+1)*self.tp, (1+self.beta**2)*self.tp+self.beta**2*self.fn+self.fp)
#             return 1./tf.cast(self.num_thresholds, tf.float32)*tf.cast(tf.argmax(afs)+1, tf.float32)
#         elif self.name_=='AUC':
#             epsilon = 1.0e-6
#             rec = tf.divide(self.tp + epsilon, self.tp + self.fn + epsilon)
#             fp_rate = tf.divide(self.fp, self.fp + self.tn + epsilon)
#             x = tf.concat([[1.], fp_rate, [0.]],0)
#             y = tf.concat([[1.], rec, [0.]],0)
#             return tf.reduce_sum(tf.multiply(x[:-1] - x[1:],
#                                            (y[:-1] + y[1:]) / 2.))
#         else: raise NotImplementedError

#     def reset_state(self):
#         self.tp.assign(tf.zeros((self.num_thresholds-1,)))
#         self.tn.assign(tf.zeros((self.num_thresholds-1,)))
#         self.fn.assign(tf.zeros((self.num_thresholds-1,)))
#         self.fp.assign(tf.zeros((self.num_thresholds-1,)))

class MultilabelMetrics(tf.keras.metrics.Metric):
    def __init__(self, threshold=None, num_thresholds=None, aggregation=None, average='micro', from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.num_thresholds = num_thresholds
        self.aggregation = aggregation
        self.average = average
        self.from_logits = from_logits
        self.built = False
        
    def build(self, y_true, y_pred):
        try:
            self.num_classes = y_pred.get_shape()[-1]
        except:
            self.num_classes = y_pred.shape[-1]
        if self.num_thresholds is not None:
            self.shape = (self.num_thresholds, self.num_classes) if self.average=='macro' else (self.num_thresholds,)
        else:
            self.shape = (self.num_classes,) if self.average=='macro' else ()
        self.tp = self.add_weight(shape=self.shape, dtype=tf.float32, name='tp', initializer='zeros')
        self.tn = self.add_weight(shape=self.shape, dtype=tf.float32, name='tn', initializer='zeros')
        self.fn = self.add_weight(shape=self.shape, dtype=tf.float32, name='fn', initializer='zeros')
        self.fp = self.add_weight(shape=self.shape, dtype=tf.float32, name='fp', initializer='zeros')
        self.built = True
        
    def decode_prediction(self, y_true, y_pred, threshold, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        y_true = tf.cast(y_true, tf.float32)

        if threshold is not None:
            y_true = tf.cast(y_true>0.5, tf.float32) #0.5 for possible pre-label smoothing
            y_pred = tf.cast(y_pred>threshold, tf.float32)

        return y_true, y_pred

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        if not self.built:
            self.build(y_true, y_pred)
        
        if self.num_thresholds is not None:
            arr = tf.TensorArray(tf.float32, self.num_thresholds, clear_after_read=True)
            for i in tf.range(self.num_thresholds):
                th = tf.cast(i+1, tf.float32)/tf.cast(self.num_thresholds+1, tf.float32)
                y_t, y_p = self.decode_prediction(y_true, y_pred, th, sample_weight)
                tps = y_t * y_p #tf.cast(tf.logical_and(y_t, y_p), tf.float32)
                tns = (1-y_t) * (1-y_p) #tf.cast(tf.logical_and(tf.logical_not(y_t), tf.logical_not(y_p)), tf.float32)
                fns = y_t * (1-y_p) #tf.cast(tf.logical_and(y_t, tf.logical_not(y_p)), tf.float32)
                fps = (1-y_t) * y_p #tf.cast(tf.logical_and(tf.logical_not(y_t), y_p), tf.float32)
                if self.average=='macro':
                    # we assume last axis always represents classes
                    tps = tf.reduce_sum(tf.reshape(tps, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(tps, self.num_classes, -1)))
                    tns = tf.reduce_sum(tf.reshape(tns, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(tns, self.num_classes, -1)))
                    fns = tf.reduce_sum(tf.reshape(fns, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(fns, self.num_classes, -1)))
                    fps = tf.reduce_sum(tf.reshape(fps, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(fps, self.num_classes, -1)))
                else:
                    tps = tf.reduce_sum(tps)
                    tns = tf.reduce_sum(tns)
                    fns = tf.reduce_sum(fns)
                    fps = tf.reduce_sum(fps)
                arr = arr.write(i, [tps, tns, fns, fps]) #(n,3)
            tps, tns, fns, fps= tf.unstack(arr.stack(), axis=1)
        else:
            y_t, y_p = self.decode_prediction(y_true, y_pred, self.threshold, sample_weight)
            tps = y_t * y_p #tf.cast(tf.logical_and(y_t, y_p), tf.float32)
            tns = (1-y_t) * (1-y_p) #tf.cast(tf.logical_and(tf.logical_not(y_t), tf.logical_not(y_p)), tf.float32)
            fns = y_t * (1-y_p) #tf.cast(tf.logical_and(y_t, tf.logical_not(y_p)), tf.float32)
            fps = (1-y_t) * y_p #tf.cast(tf.logical_and(tf.logical_not(y_t), y_p), tf.float32)
            if self.average=='macro':
                # we assume last axis always represents classes
                tps = tf.reduce_sum(tf.reshape(tps, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(tps, self.num_classes, -1)))
                tns = tf.reduce_sum(tf.reshape(tns, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(tns, self.num_classes, -1)))
                fns = tf.reduce_sum(tf.reshape(fns, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(fns, self.num_classes, -1)))
                fps = tf.reduce_sum(tf.reshape(fps, (-1,self.num_classes)), axis=0)#tf.map_fn(tf.reduce_sum, tf.stack(tf.unstack(fps, self.num_classes, -1)))
            else:
                tps = tf.reduce_sum(tps)
                tns = tf.reduce_sum(tns)
                fns = tf.reduce_sum(fns)
                fps = tf.reduce_sum(fps)
        self.tp.assign_add(tps)
        self.tn.assign_add(tns)
        self.fn.assign_add(fns)
        self.fp.assign_add(fps)
    
    def aggregate_result(self,result,aggregation):
        if self.average == 'macro':
            result = tf.reduce_mean(result, axis=-1)
        if aggregation is not None:
            if aggregation == 'mean':
                result = tf.reduce_mean(result)
            elif aggregation == 'max':
                result = tf.reduce_max(result)
            elif aggregation == 'min':
                result = tf.reduce_min(result)
            elif aggregation == 'thr':
                result = tf.cast(tf.argmax(result)+1, tf.float32)/tf.cast(self.num_thresholds+1, tf.float32)
            else:
                raise NotImplementedError
        else:
            result = tf.reduce_mean(result)
        return result
        
    def result(self):
        raise NotImplementedError

    def reset_state(self):
        reset_value = tf.zeros(self.shape, dtype=tf.float32)
        tf.keras.backend.batch_set_value([(v, reset_value) for v in self.variables])

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()

class Acc(MultilabelMetrics):
    def __init__(self, threshold=0.5, num_thresholds=None, aggregation=None, average='micro', from_logits=False, **kwargs):
        super().__init__(threshold=threshold, num_thresholds=num_thresholds, aggregation=aggregation, average=average, from_logits=from_logits, **kwargs)
    def result(self):
        result = tf.math.divide_no_nan(self.tp+self.tn, self.tp+self.fp+self.fn+self.tn)
        return self.aggregate_result(result, self.aggregation)

class Precision(MultilabelMetrics):
    def __init__(self, threshold=0.5, num_thresholds=None, aggregation=None, average='micro', from_logits=False, **kwargs):
        super().__init__(threshold=threshold, num_thresholds=num_thresholds, aggregation=aggregation, average=average, from_logits=from_logits, **kwargs)
    def result(self):
        result = tf.math.divide_no_nan(self.tp, self.tp+self.fp)
        return self.aggregate_result(result, self.aggregation)

class Recall(MultilabelMetrics):
    def __init__(self, threshold=0.5, num_thresholds=None, aggregation=None, average='micro', from_logits=False, **kwargs):
        super().__init__(threshold=threshold, num_thresholds=num_thresholds, aggregation=aggregation, average=average, from_logits=from_logits, **kwargs)
    def result(self):
        result = tf.math.divide_no_nan(self.tp, self.tp+self.fn)
        return self.aggregate_result(result, self.aggregation)
    
class Fbeta(MultilabelMetrics):
    def __init__(self, beta=1, threshold=0.5, num_thresholds=None, aggregation=None, average='micro', from_logits=False, **kwargs):
        super().__init__(threshold=threshold, num_thresholds=num_thresholds, aggregation=aggregation, average=average, from_logits=from_logits, **kwargs)
        self.beta = beta
    def result(self):
        result = tf.math.divide_no_nan((self.beta**2+1)*self.tp, (1+self.beta**2)*self.tp+self.beta**2*self.fn+self.fp)
        return self.aggregate_result(result, self.aggregation)
    
class IoU(MultilabelMetrics):
    def __init__(self, threshold=0.5, num_thresholds=None, aggregation=None, average='micro', from_logits=False, **kwargs):
        super().__init__(threshold=threshold, num_thresholds=num_thresholds, aggregation=aggregation, average=average, from_logits=from_logits, **kwargs)
    def result(self):
        result = tf.math.divide_no_nan(self.tp , self.tp + self.fp + self.fn)
        return self.aggregate_result(result, self.aggregation)
    
class AUC(MultilabelMetrics):
    def __init__(self, num_thresholds=50, from_logits=False, **kwargs):
        super().__init__(threshold=None, num_thresholds=num_thresholds, aggregation=None, average='micro', from_logits=from_logits, **kwargs)
    def result(self):
        epsilon = 1.0e-6
        rec = tf.divide(self.tp + epsilon, self.tp + self.fn + epsilon)
        fp_rate = tf.divide(self.fp, self.fp + self.tn + epsilon)
        x = tf.concat([[1.], fp_rate, [0.]],0)
        y = tf.concat([[1.], rec, [0.]],0)
        return tf.reduce_sum(tf.multiply(x[:-1] - x[1:],
                                       (y[:-1] + y[1:]) / 2.))
