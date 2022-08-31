# imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf


class Server:

    def __init__(self, num_data_holder_parties, tf_seed):
        self.model = None
        self.define_model()
        self.global_model_parameters = None
        self.global_grads = None

        self.num_data_holder_parties = num_data_holder_parties

        tf.random.set_seed(tf_seed)

    def define_model(self):
        """ This function generates the NN model"""

        self.model = keras.Sequential([
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

        self.model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 'binary_crossentropy',
            metrics=['accuracy'],
        )

    def aggregate_grads(self, grads_list):
        """ This function aggregates the received gradients from data holder parties"""

        parties_grads_list = list(grads_list.keys())
        self.global_grads = grads_list[parties_grads_list[0]]
        for i in range(1, self.num_data_holder_parties):
            temp = grads_list[parties_grads_list[i]]
            for j in range(0, len(self.global_grads)):
                self.global_grads[j] += temp[j]

    def interface_pipeline(self, grads_list):
        """ This can be called by interface """

        # receive grads from data holder parties
        # aggregate grads
        self.aggregate_grads(grads_list)
        # update model based on received grads
        self.model.optimizer.apply_gradients(zip(self.global_grads, self.model.trainable_weights))
        # update self.global_model_parameters
        self.global_model_parameters = self.model.get_weights()
        # share/send global_model_parameters to interface/data holder parties
        return self.global_model_parameters

