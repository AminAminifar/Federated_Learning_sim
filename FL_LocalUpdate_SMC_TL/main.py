import server_party_class
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf

from get_train_test_data_alter import generate_train_test_data, load_data_for_all_model
from model import get_all_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

final_score_other = []

# generate test and train data and save those in test_train_data folder
print("1-generate test and train data and save those in test_train_data folder")
generate_train_test_data(path="dataset")

# parties uses data which generated in previous section
import generate_parties

# load data for training a base modek
print("2-load data for training the all model")
x_train_other, x_val_other, x_test_other, y_train_other, y_val_other, y_test_other, x_test_transfer, y_test_transfer = load_data_for_all_model(path="test_train_data")

# print(x_train_other.shape)
# print(y_train_other.shape)

# train the all model
print("3-load all model")
model_base, model_head = get_all_model(head_size=32)

model_other = tf.keras.Sequential(
        [model_base, model_head]
)
model_other.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

# checkpoint_path = "model_base/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_weights_only=True,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True)

print("4-train all model")
history = model_other.fit(x_train_other, y_train_other, validation_data=(x_val_other, y_val_other), batch_size=16, epochs=1, verbose=1)

print("5-evaluate the all model")
score = model_other.evaluate(x_test_transfer, y_test_transfer, verbose=0)

print("all_other_score:", score)
final_score_other.append(score[1])

print("save model base")
model_base.save("model_base\model_base.h5")

print("6-preparing model for transfer learning")
# model_base.trainable = False 
# model_head.trainable = True 

model_head.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 'binary_crossentropy',
    metrics=['accuracy'],
)

tf_seed = 0
num_data_holder_parties = 5
num_local_updates = 3
scenario = 2

# generate/instantiate parties
print("7-generate/instantiate parties")
data_holder_parties_all = generate_parties.generate_parties(num_data_holder_parties=num_data_holder_parties,
                                                            tf_seed=tf_seed,
                                                            num_local_updates=num_local_updates,
                                                            scenario=scenario)

print("8-generate/instantiate mediator")
server_party = server_party_class.Server(num_data_holder_parties=num_data_holder_parties,
                                         tf_seed=tf_seed)

# repeat training process (as the interface)
global_model_parameters = None
num_epoch = 10  # 100
test_loss, test_acc = np.zeros(num_epoch), np.zeros(num_epoch)
for epoch in range(num_epoch):
    print("iteration", epoch)
    try:
        model_parameters_dict
        model_parameters_dict.clear()
    except NameError:
        pass
    model_parameters_dict = {}
    for data_holder_i in range(0, num_data_holder_parties):
        if global_model_parameters is None:
            model_parameters_dict[data_holder_i] = data_holder_parties_all[data_holder_i].interface_pipeline()
        else:
            model_parameters_dict[data_holder_i] = \
                data_holder_parties_all[data_holder_i].interface_pipeline(global_model_parameters)
    global_model_parameters = server_party.interface_pipeline(model_parameters_dict)


    # for evaluation purpose only

    model_head.set_weights(global_model_parameters)
    model_other = tf.keras.Sequential(
        [model_base, model_head]
    )
    model_other.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 'binary_crossentropy',
        metrics=['accuracy'],
    )
    test_loss[epoch], test_acc[epoch] = model_other.evaluate(x_test_transfer, y_test_transfer, verbose=2)

# # plot evaluation results
# fig,  axs = plt.subplots(1, 2)
# axs[0].set_title("test_loss")
# axs[0].plot(range(1, num_epoch+1), test_loss)
# axs[0].set(xlabel='epoch', ylabel='test_loss')

# axs[1].set_title("test_acc")
# axs[1].plot(range(1, num_epoch+1), test_acc)
# axs[1].set(xlabel='epoch', ylabel='test_acc')
# plt.show()



