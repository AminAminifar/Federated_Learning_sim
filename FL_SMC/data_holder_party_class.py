# imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf
import SMC_functions


class Party:

    def __init__(self, data, data_labels, tf_seed, num_parties, party_id, scenario):
        self.data = data
        self.data_labels = data_labels
        # Instantiate a loss function.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model = self.define_model()
        self.grads = None
        self.tf_seed = tf_seed
        self.SMC_tools = SMC_functions.SMCtools(num_parties=num_parties, party_id=party_id,
                                                num_participating_parties=num_parties,
                                                secure_aggregation_parameter_k=num_parties-1,
                                                scenario=scenario)


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
            loss=self.loss_fn,  # 'binary_crossentropy', categorical_crossentropy
            metrics=['accuracy'],
        )
        return model

    def calculate_gradients(self, x, y):
        """ This function calculates gradients for one round of feedforward and back propagation """

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
        else:
            tf.random.set_seed(self.tf_seed)
        # calculate gradients
        self.calculate_gradients(self.data, self.data_labels)
        # share grads
        return self.grads




