from pathlib import Path
import gzip
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pandas as pd
import os

model_base = keras.Sequential([
    # base
    # layers.Conv2D(64, (5, 5), padding="same", activation="relu", input_shape=(28, 28, 1)),
    # layers.MaxPool2D(pool_size=(2, 2)),
    #
    # layers.Conv2D(128, (5, 5), padding="same", activation="relu"),
    # layers.MaxPool2D(pool_size=(2, 2)),
    #
    # layers.Conv2D(256, (5, 5), padding="same", activation="relu"),
    # layers.MaxPool2D(pool_size=(2, 2)),

    # base
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform',
                      input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

])

model_head = keras.Sequential([

    # Head
    # layers.Flatten(),
    # layers.Dense(256, activation='relu'),
    # layers.Dense(10),

    # Head
    layers.Flatten(),
    layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
    layers.Dense(10, activation='softmax')

])

model = keras.Sequential(
        [model_base, model_head]
    )



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
train_data_all = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
train_data_all = train_data_all.reshape(num_images, image_size, image_size, 1)
train_data = train_data_all[:int(np.round(num_images*.5)),:,:,:]


# image = np.asarray(train_data[1]).squeeze()
# plt.imshow(image)  # , cmap="Greys"
# plt.show()


# labels
f = gzip.open(data_dir/'train-labels-idx1-ubyte.gz','r')
f.read(8)

buf = f.read(num_images)
train_labels_all = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
# print(labels)
train_labels = train_labels_all[:int(np.round(num_images*.5))]



history = model.fit(
    train_data, train_labels,
    validation_data=(test_data,test_labels),
    epochs=10,  # 10
    verbose=1
)

# history_frame = pd.DataFrame(history.history)
# history_frame.loc[:, ['loss', 'val_loss']].plot()
# history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
# plt.show()

checkpoint_path = "model_base/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model_base.save_weights(checkpoint_path.format(epoch=0))
print("Model Base was saved.")
