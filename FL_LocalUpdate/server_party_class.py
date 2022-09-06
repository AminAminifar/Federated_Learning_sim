# imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf


class Server:

    def __init__(self, num_data_holder_parties, tf_seed):
        # self.model = None
        # self.define_model()
        self.global_model_parameters = None

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

    def aggregate_model_parameters(self, model_parameters_dict):
        """ This function aggregates the received model parameters from data holder parties"""

        parties_model_parameters_list = list(model_parameters_dict.keys())
        self.global_model_parameters = model_parameters_dict[parties_model_parameters_list[0]]
        for i in range(1, self.num_data_holder_parties):
            temp = model_parameters_dict[parties_model_parameters_list[i]]
            for j in range(0, len(self.global_model_parameters)):
                self.global_model_parameters[j] += temp[j]

        for j in range(0, len(self.global_model_parameters)):
            self.global_model_parameters[j] /= self.num_data_holder_parties

    def interface_pipeline(self, model_parameters_dict):
        """ This can be called by interface """

        # receive model parameters from data holder parties
        # aggregate model parameters
        self.aggregate_model_parameters(model_parameters_dict)
        # share/send global_model_parameters to interface/data holder parties
        return self.global_model_parameters

