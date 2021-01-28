src_train_dirname = "./data/gonczy/few_train"
src_val_dirname = "./data/gonczy/few_test"
tgt_train_dirname = "./data/wddd/z=31/few"
tgt_val_dirname = "./data/wddd/z=33/few"

pretrain = not True
adapt = True
evaluate = False

# params for dataset and data loader
pretrain_batch_size = 8
adapt_batch_size = 8

# params for setting up models
snapshot_path = "./snapshots"
encoder_filename = "encoder.pt"
decoder1_filename = "decoder1.pt"
decoder2_filename = "decoder2.pt"
d_input_dims = 1024
d_hidden_dims = 0
d_output_dims = 2

# params for training network
input_channel = 1
num_class = 2
image_size = 128
num_iter = 50
pre_num_epochs = 100
adapt_num_epochs = 100
num_k = 3

# params for optimizing models
pre_learning_rate = 1e-3
e_learning_rate = 1e-3
c_learning_rate = 1e-3
beta1 = 0.5
beta2 = 0.9

max_discrepancy = False
