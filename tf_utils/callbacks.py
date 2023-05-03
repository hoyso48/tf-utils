import tensorflow as tf

class Snapshot(tf.keras.callbacks.Callback):
    
    def __init__(self,save_name,snapshot_epochs=[]):
        super().__init__()
        self.snapshot_epochs = snapshot_epochs
        self.save_name = save_name
        
    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        if epoch in self.snapshot_epochs: # your custom condition         
            self.model.save_weights(f"{self.save_name}-epoch{epoch}.h5")
        self.model.save_weights(f"{self.save_name}-last.h5")
        
        
class SWA(tf.keras.callbacks.Callback):

    def __init__(self,save_name,swa_epochs=[],strategy=None,train_ds=None,valid_ds=None,train_steps=1000,valid_steps=None):
        super().__init__()
        self.swa_epochs = swa_epochs
        self.swa_weights = None
        self.save_name = save_name
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.train_steps = train_steps
        self.valid_steps = valid_steps
        self.strategy = strategy
    
    @tf.function
    def train_step(self, iterator):
        """The step function for one training step."""
        def step_fn(inputs):
            """The computation to run on each device."""
            x,y = inputs
            _ = self.model(x, training=True)

        for x in iterator:
            self.strategy.run(step_fn, args=(x,))
            
    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.swa_epochs:   
            if self.swa_weights is None:
                self.swa_weights = self.model.get_weights()
            else:
                w = self.model.get_weights()
                for i in range(len(self.swa_weights)):
                    self.swa_weights[i] += w[i]                    
    
    def on_train_end(self, logs=None):
        if len(self.swa_epochs):
          print('applying SWA...')
          for i in range(len(self.swa_weights)):
              self.swa_weights[i] = self.swa_weights[i]/len(self.swa_epochs)
          self.model.set_weights(self.swa_weights)
          if self.train_ds is not None: #for the re-calculation of running mean and var
              self.train_step(self.train_ds.take(self.train_steps))
          print(f'save SWA weights to {self.save_name}-SWA.h5')
          self.model.save_weights(f"{self.save_name}-SWA.h5")
          if self.valid_ds is not None:
              self.model.evaluate(self.valid_ds, steps=self.valid_steps)
