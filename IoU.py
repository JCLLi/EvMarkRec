import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class CustomIoU(tf.keras.metrics.Metric):
    def __init__(self, name='custom_iou', **kwargs):
        super(CustomIoU, self).__init__(name=name, **kwargs)
        self.iou_metric = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0, 1])
        self.iou = self.add_weight(name='iou', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.iou_metric.update_state(y_true, y_pred, sample_weight)
        self.iou.assign(self.iou_metric.result())

    def result(self):
        return self.iou

    def reset_states(self):
        self.iou_metric.reset_states()
        self.iou.assign(0.0)

class CustomIoUCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        logs['custom_iou'] = self.model.metrics[0].result().numpy()