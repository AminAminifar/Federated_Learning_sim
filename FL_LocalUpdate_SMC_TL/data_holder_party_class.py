# imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf
import SMC_functions
import os

class Party:

    def __init__(self, data, data_labels, tf_seed, num_local_updates, num_parties, party_id, scenario):
        self.data_raw = data
        self.data = None
        self.predict_with_model_base()
        self.data_labels = data_labels
        # Instantiate a loss function.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.grads = None
        self.num_local_updates = num_local_updates
        self.tf_seed = tf_seed
        self.model = self.define_model()
        self.SMC_tools = SMC_functions.SMCtools(num_parties=num_parties, party_id=party_id,
                                                num_participating_parties=num_parties,
                                                secure_aggregation_parameter_k=num_parties - 1,
                                                scenario=scenario)

    def predict_with_model_base(self):

        model_base = keras.models.load_model("model_base/model_base.h5")
        # model_base = keras.Sequential([
        #     # base
        #     layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform',
        #                   input_shape=(28, 28, 1)),
        #     layers.MaxPooling2D((2, 2)),
        # ])
        # checkpoint_path = "model_base/cp-{epoch:04d}.ckpt"
        # checkpoint_dir = os.path.dirname(checkpoint_path)

        # latest = tf.train.latest_checkpoint(checkpoint_dir)
        # model_base.load_weights(latest)

        self.data = model_base.predict(self.data_raw)

    def define_model(self):
        """ This function generates the NN model"""

        # def my_init(shape, dtype=None):
        #     return tf.keras.backend.random_normal(shape, dtype=dtype, seed=self.tf_seed)
        # initializer = tf.keras.initializers.GlorotUniform(seed=self.tf_seed)
        # initializer = tf.keras.initializers.Zeros()

        model_head = tf.keras.models.Sequential([tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.L1(0.001)),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation('relu'),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(2, activation=tf.nn.softmax)])

        model = model_head

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

    def locally_update_model(self):
        """ This function updates the model one to several times based on local data
         and returns the updated parameters """

        for i in range(0, self.num_local_updates):
            # calculate gradients
            self.calculate_gradients(self.data, self.data_labels)
            # update model based on calculated grads
            self.model.optimizer.apply_gradients(zip(self.grads, self.model.trainable_weights))

        # return locally updated model parameters
        return self.model.get_weights()

    def interface_pipeline(self, global_model_parameters=None):
        """ This can be called by server/interface """

        if global_model_parameters is not None:
            #  receive global model parameters
            # update local model by global_model_parameters
            self.model.set_weights(global_model_parameters)
        # locally update the model
        self.locally_update_model()
        model_parameters = self.locally_update_model()
        # mask updated model parameters
        masked_model_parameters = self.SMC_tools.mask(model_parameters)
        # share masked updated model parameters
        return masked_model_parameters




