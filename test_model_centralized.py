from pathlib import Path
import gzip
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pandas as pd

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

    # # new
    # layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform',
    #               input_shape=(28, 28, 1)),
    # layers.MaxPooling2D((2, 2)),
    # layers.Flatten(),
    # layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
    # layers.Dense(10, activation='softmax')
])



model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 'binary_crossentropy', categorical_crossentropy
    metrics=['accuracy'],  # accuracy, binary_accuracy
)




# load test data
data_dir = Path('C:/Amin/Workspace/Data/fashion mnist')

# load test data
f = gzip.open(data_dir/'t10k-images-idx3-ubyte.gz','r')
image_size = 28
num_images = 10000


f.read(16)
buf = f.read(image_size * image_size * num_images)
test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
test_data = test_data.reshape(num_images, image_size, image_size, 1)

# image = np.asarray(test_data[2]).squeeze()
# plt.imshow(image, cmap="Greys")
# plt.show()

# labels
f = gzip.open(data_dir/'t10k-labels-idx1-ubyte.gz','r')
f.read(8)

buf = f.read(num_images)
test_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
# print(labels)


# load train data
f = gzip.open(data_dir/'train-images-idx3-ubyte.gz','r')
image_size = 28
num_images = 60000


f.read(16)
buf = f.read(image_size * image_size * num_images)
train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
train_data = train_data.reshape(num_images, image_size, image_size, 1)

# image = np.asarray(train_data[2]).squeeze()
# plt.imshow(image, cmap="Greys")
# plt.show()

# labels
f = gzip.open(data_dir/'train-labels-idx1-ubyte.gz','r')
f.read(8)

buf = f.read(num_images)
train_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
# print(labels)


# def prep_pixels(train, test):
# 	# convert from integers to floats
# 	train_norm = train.astype('float32')
# 	test_norm = test.astype('float32')
# 	# normalize to range 0-1
# 	train_norm = train_norm / 255.0
# 	test_norm = test_norm / 255.0
# 	# return normalized images
# 	return train_norm, test_norm
#
# train_data, test_data = prep_pixels(train_data, test_data)

history = model.fit(
    train_data, train_labels,
    validation_data=(test_data,test_labels),
    epochs=10,
    verbose=1
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()