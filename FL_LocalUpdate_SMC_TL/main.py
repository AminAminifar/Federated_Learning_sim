import sys
import server_party_class
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# from get_train_test_data_alter import generate_train_test_data, load_data_for_all_model
from get_train_test_data_alter import generate_train_test_data, load_data_for_all_model
from model import get_all_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def run(argv):

    iteration = int(argv[1]) # 100
    
    num_data_holder_parties = int(argv[2]) # 5
    num_local_updates = int(argv[3]) # 3
    scenario = int(argv[4]) # 2

    tf_seed = 0

    final_acc = []
    final_pre = []
    final_re = []
    final_f = []

    for i in range(iteration):
        print("iteration: {}".format(i))
        # generate test and train data and save those in test_train_data folder
        print("1-generate test and train data and save those in test_train_data folder")
        generate_train_test_data(path="dataset")

        # parties uses data which generated in previous section
        import generate_parties

        # load data for training a base modek
        print("2-load data for training the all model")
        x_train_other, x_val_other, x_test_other, y_train_other, y_val_other, y_test_other, x_test_transfer, y_test_transfer = load_data_for_all_model(path="test_train_data")

        # train the all model
        print("3-load all model")
        model_base, model_head = get_all_model(head_size=32)

        model_other = tf.keras.Sequential(
                [model_base, model_head]
        )
        model_other.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

        print("4-train all model")
        history = model_other.fit(x_train_other, y_train_other, validation_data=(x_val_other, y_val_other), batch_size=16, epochs=1, verbose=1)

        print("5-evaluate the all model")
        score = model_other.evaluate(x_test_transfer, y_test_transfer, verbose=0)

        print("all_other_score:", score)

        print("save model base")
        model_base.save("model_base\model_base.h5")

        print("6-preparing model for transfer learning")

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
        num_epoch = 1  # 100
        test_loss, test_acc = np.zeros(num_epoch), np.zeros(num_epoch)
        for epoch in range(num_epoch):
            print("epoch", epoch)
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

        model_head.set_weights(global_model_parameters)
        model_other = tf.keras.Sequential(
            [model_base, model_head]
        )
        model_other.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 'binary_crossentropy',
            metrics=['accuracy'],
        )
        y_pred = model_other.predict(x_test_transfer)
        y_pred = np.argmax(y_pred, axis=1)

        final_acc.append(accuracy_score(y_test_transfer, y_pred))
        final_pre.append(precision_score(y_test_transfer, y_pred))
        final_re.append(recall_score(y_test_transfer, y_pred))
        final_f.append(f1_score(y_test_transfer, y_pred))

        print('accuracy_score :', accuracy_score(y_test_transfer, y_pred))
        print('recall_score: ', recall_score(y_test_transfer, y_pred))
        print('precision_score: ', precision_score(y_test_transfer, y_pred))
        print('f1_score: ', f1_score(y_test_transfer, y_pred))

    print("Final Accuracy: ", np.mean(final_acc))
    print("Final Precision: ", np.mean(final_pre))
    print("Final Recall: ", np.mean(final_re))
    print("Final F1: ", np.mean(final_f))

if __name__  == "__main__":
    run(sys.argv)

