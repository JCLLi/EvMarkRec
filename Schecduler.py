import tensorflow as tf
import math



class LinCos(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr, end_lr, warmup_steps, total_steps, start_steps):
        super(LinCos, self).__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_step = start_steps

    def __call__(self, step):
        step += self.start_step

        def lr_warmup():
            return tf.cast(step / self.warmup_steps * self.start_lr, tf.float32)

        def lr_cosine_decay():
            cosine_step = tf.cast(step - self.warmup_steps, tf.float32)
            cosine_steps_total = tf.cast(self.total_steps - self.warmup_steps, tf.float32)
            cosine_decay = 0.5 * (1 + tf.math.cos(tf.constant(math.pi) * cosine_step / cosine_steps_total))
            return self.end_lr + (self.start_lr - self.end_lr) * cosine_decay
        
        result = tf.cond(step < self.warmup_steps, lr_warmup, lr_cosine_decay)
        print("LR: " + str(result))
        return result
    
    def get_config(self):
        return {
            'start_lr': self.start_lr,
            'end_lr': self.end_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'start_step': self.start_step
        }
    

class ConCos(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr, end_lr, constant_steps, total_steps, start_steps):
        super(ConCos, self).__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.constant_steps = constant_steps
        self.total_steps = total_steps
        self.start_step = start_steps

    def __call__(self, step):
        step += self.start_step

        def lr_constant():
            return tf.cast(self.start_lr, tf.float32)

        def lr_cosine_decay():
            cosine_step = tf.cast(step - self.constant_steps, tf.float32)
            cosine_steps_total = tf.cast(self.total_steps - self.constant_steps, tf.float32)
            cosine_decay = 0.5 * (1 + tf.math.cos(tf.constant(math.pi) * cosine_step / cosine_steps_total))
            return self.end_lr + (self.start_lr - self.end_lr) * cosine_decay
        
        result = tf.cond(step < self.constant_steps, lr_constant, lr_cosine_decay)
        # print("LR: " + str(result))
        return result
    
    def get_config(self):
        return {
            'start_lr': self.start_lr,
            'end_lr': self.end_lr,
            'constant_steps': self.constant_steps,
            'total_steps': self.total_steps,
            'start_step': self.start_step
        }