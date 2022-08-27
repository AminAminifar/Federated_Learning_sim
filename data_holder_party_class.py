# imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf


class Party:

    def __init__(self, data, data_labels, tf_seed):
        self.data = data
        self.data_labels = data_labels
        # Instantiate a loss function.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model = self.define_model()
        self.grads = None
        tf.random.set_seed(tf_seed)

    def define_model(self):
        """ This function generates the NN model"""

        model = keras.Sequential([
            layers.InputLayer(input_shape=[28, 28]),

            #             # Data Augmentation
            #             preprocessing.RandomFlip('horizontal'),
            #             preprocessing.RandomContrast(0.5),

            #             # Conv
            #             layers.BatchNormalization(renorm=True),
            #             layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
            #             layers.MaxPool2D(),

            # Head
            layers.BatchNormalization(renorm=True),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10),

        ])

        model.compile(
            optimizer='adam',
            loss=self.loss_fn,  # 'binary_crossentropy',
            metrics=['binary_accuracy'],
        )
        return model

    def calculate_gradients(self, x, y):
        """ This function calculate gradients for one round of feedforward and back propagation """

        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
        self.grads = tape.gradient(loss_value, self.model.trainable_weights)

    def interface_pipeline(self, global_model_parameters=None):
        """ This can be called by server/interface """

        if global_model_parameters is not None:
            #  receive global model parameters
            # update local model by global_model_parameters
            self.model.set_weights(global_model_parameters)
        # calculate gradients
        self.calculate_gradients(self.data, self.data_labels)
        # share grads
        return self.grads




