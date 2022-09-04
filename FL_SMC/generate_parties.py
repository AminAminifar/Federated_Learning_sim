# imports
import gzip
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle
import data_holder_party_class


# load data
data_dir = Path('C:/Amin/Workspace/Data/fashion mnist')

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

# data information
num_train_records = 60000
train_data_record_indices = range(0, num_train_records)
train_data_record_indices_shuffled = shuffle(train_data_record_indices, random_state=0)


def generate_parties(num_data_holder_parties, tf_seed, scenario):
    """ This function generates data holder parties and return a list of generated parties"""

    # indices of training data samples for each data holder party
    chunk_indices = np.array_split(train_data_record_indices_shuffled, num_data_holder_parties)

    parties = []
    for i in range(num_data_holder_parties):
        parties.append(data_holder_party_class.Party(data=train_data[chunk_indices[i]],
                                                     data_labels=train_labels[chunk_indices[i]],
                                                     tf_seed=tf_seed,
                                                     num_parties=num_data_holder_parties,
                                                     party_id=i,
                                                     scenario=scenario))

    print("A list of parties have been created.\n\
    Data samples randomly assigned to each party.\n\
    The number of samples for each party is almost the same.")

    return parties
