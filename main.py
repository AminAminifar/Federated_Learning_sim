import data_holder_party_class
import server_party_class
import generate_parties

tf_seed = 0
num_data_holder_parties = 2

# generate/instantiate parties
data_holder_parties_all = generate_parties.generate(num_data_holder_parties=num_data_holder_parties,
                                                    tf_seed=tf_seed)
server_party = server_party_class.Server(num_data_holder_parties=num_data_holder_parties,
                                         f_seed=tf_seed)


# repeat training process (as the interface)
Grads_list = []
global_model_parameters = None
for iteration in range(1):
    print("iteration", iteration)
    for data_holder_i in range(0, num_data_holder_parties):
        if global_model_parameters is None:
            Grads_list.append(data_holder_parties_all[data_holder_i].interface_pipeline())
        else:
            Grads_list.append(data_holder_parties_all[data_holder_i].interface_pipeline(global_model_parameters))
        global_model_parameters = server_party.interface_pipeline(Grads_list)

