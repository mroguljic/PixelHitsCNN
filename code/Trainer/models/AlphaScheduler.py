import tensorflow as tf

class AlphaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, initial_alpha=1.0, decay_rate=0.04):
        self.total_epochs = total_epochs
        self.initial_alpha = initial_alpha
        self.decay_rate = decay_rate

    def on_epoch_begin(self, epoch, logs=None):
        # Linearly decay alpha from 1.0 to 0.0
        alpha = max(self.initial_alpha - (self.decay_rate * epoch), 0.0)
        self.alpha = alpha
        print(f"Epoch {epoch+1}: Setting alpha to {alpha:.4f}")
        self.model.alpha.assign(alpha)
