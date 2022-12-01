import tensorflow as tf
import matplotlib.pyplot as plt

class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''
    Unified single-cycle learning rate scheduler for tensorflow.
    2022 Hoyeol Sohn <hoeyol0730@gmail.com>
    '''
    def __init__(self,
                lr=1e-4,
                epochs=10,
                steps_per_epoch=100,
                steps_per_update=1,
                resume_epoch=0,
                decay_epochs=10,
                sustain_epochs=0,
                warmup_epochs=0,
                lr_start=0,
                lr_min=0,
                warmup_type='linear',
                decay_type='cosine',
                **kwargs):
        
        super().__init__(**kwargs)
        self.lr = float(lr)
        self.epochs = float(epochs)
        self.steps_per_update = float(steps_per_update)
        self.resume_epoch = float(resume_epoch)
        self.steps_per_epoch = float(steps_per_epoch)
        self.decay_epochs = float(decay_epochs)
        self.sustain_epochs = float(sustain_epochs)
        self.warmup_epochs = float(warmup_epochs)
        self.lr_start = float(lr_start)
        self.lr_min = float(lr_min)
        self.decay_type = decay_type
        self.warmup_type = warmup_type
        

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        total_steps = self.epochs * self.steps_per_epoch
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        sustain_steps = self.sustain_epochs * self.steps_per_epoch
        decay_steps = self.decay_epochs * self.steps_per_epoch

        if self.resume_epoch > 0:
            step = step + self.resume_epoch * self.steps_per_epoch

        step = tf.cond(step > decay_steps, lambda :decay_steps, lambda :step)
        step = tf.math.truediv(step, self.steps_per_update) * self.steps_per_update

        warmup_cond = step < warmup_steps
        decay_cond = step >= (warmup_steps + sustain_steps)
        
        if self.warmup_type == 'linear':
            lr = tf.cond(warmup_cond, lambda: tf.math.divide_no_nan(self.lr-self.lr_start , warmup_steps) * step + self.lr_start, lambda: self.lr)
        elif self.warmup_type == 'exponential':
            factor = tf.pow(self.lr_start, 1/warmup_steps)
            lr = tf.cond(warmup_cond, lambda: (self.lr - self.lr_start) * factor**(warmup_steps - step) + self.lr_start, lambda: self.lr)
        elif self.warmup_type == 'cosine':
            lr = tf.cond(warmup_cond, lambda: 0.5 * (self.lr - self.lr_start) * (1 + tf.cos(3.14159265359 * (warmup_steps - step)  / warmup_steps)) + self.lr_start, lambda:self.lr)
        else:
            raise NotImplementedError
                    
        
        if self.decay_type == 'linear':
            lr = tf.cond(decay_cond, lambda: self.lr + (self.lr_min-self.lr)/(decay_steps - warmup_steps - sustain_steps)*(step - warmup_steps - sustain_steps), lambda:lr)
        elif self.decay_type == 'exponential':
            factor = tf.pow(self.lr_min, 1/(decay_steps - warmup_steps - sustain_steps))
            lr = tf.cond(decay_cond, lambda: (self.lr - self.lr_min) * factor**(step - warmup_steps - sustain_steps) + self.lr_min, lambda:lr)
        elif self.decay_type == 'cosine':
            lr = tf.cond(decay_cond, lambda: 0.5 * (self.lr - self.lr_min) * (1 + tf.cos(3.14159265359 * (step - warmup_steps - sustain_steps) / (decay_steps - warmup_steps - sustain_steps))) + self.lr_min, lambda:lr)
        else:
            raise NotImplementedError
            
        return lr

    def plot(self):
        step = max(1, int(self.epochs*self.steps_per_epoch)//1000) #1 for total_steps < 1000, total_steps//1000 else
        eps = list(range(0,int(self.epochs*self.steps_per_epoch),step))
        learning_rates = [self(x) for x in eps]
        plt.scatter(eps,learning_rates,2)
        plt.show()
       

class ListedLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, schedules, steps_per_epoch=100, update_per_epoch=1, **kwargs):
        super().__init__(**kwargs)
        self.schedules = schedules
        self.steps_per_epoch = float(steps_per_epoch)
        self.update_per_epoch = float(update_per_epoch)
        for s in self.schedules:
            s.steps_per_epoch = float(steps_per_epoch)
            s.update_per_epoch = float(update_per_epoch)
        self.restart_epochs = tf.math.cumsum([s.epochs for s in self.schedules])
        self.epochs = self.restart_epochs[-1]
        self.global_steps = tf.math.cumsum([s.epochs * s.steps_per_epoch for s in self.schedules])
        
        # print(self.fns)
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        idx = tf.searchsorted(self.global_steps, [step+1])[0]
        global_steps = tf.concat([[0],self.global_steps],0)
        # fns = [lambda: self.schedules[i].__call__(step-global_steps[i]) for i in range(len(self.schedules))]
        fns = [(lambda x: (lambda: self.schedules[x].__call__(step-global_steps[x])))(i) for i in range(len(self.schedules))]
        r = tf.switch_case(idx, branch_fns=fns)
        return r

    def plot(self):
        step = max(1, int(self.epochs*self.steps_per_epoch)//1000) #1 for total_steps < 1000, total_steps//1000 else
        eps = list(range(0,int(self.epochs*self.steps_per_epoch),step))
        learning_rates = [self(x) for x in eps]
        plt.scatter(eps,learning_rates,2)
        plt.show()
