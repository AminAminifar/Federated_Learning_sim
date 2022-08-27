import server_party_class
import generate_parties
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_seed = 0
num_data_holder_parties = 2

# generate/instantiate parties
data_holder_parties_all = generate_parties.generate_parties(num_data_holder_parties=num_data_holder_parties,
                                                            tf_seed=tf_seed)
server_party = server_party_class.Server(num_data_holder_parties=num_data_holder_parties,
                                         tf_seed=tf_seed)


# repeat training process (as the interface)
global_model_parameters = None
for iteration in range(1):
    print("iteration", iteration)
    try:
        Grads_dict
        Grads_dict.clear()
    except NameError:
        pass
    Grads_dict = {}
    for data_holder_i in range(0, num_data_holder_parties):
        if global_model_parameters is None:
            Grads_dict[data_holder_i] = data_holder_parties_all[data_holder_i].interface_pipeline()
        else:
            Grads_dict[data_holder_i] = \
                data_holder_parties_all[data_holder_i].interface_pipeline(global_model_parameters)
    global_model_parameters = server_party.interface_pipeline(Grads_dict)
