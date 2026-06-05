# 2D coronal slice training - lightweight baseline
# Training on coronal slices for different anatomical view

# Data settings
input_shape = (256, 256)  # 2D coronal slices
batch_size = 8
slice_mode = "cor"  # coronal slices

# Model architecture (lightweight)
num_stages = 5
base_chs = 32
dropout = 0.1

# Training settings
nb_epochs = 1000
learning_rate = 1e-2
weight_decay = 3e-5
optimizer = "SGD"
momentum = 0.99
scheduler = "PolyLR"
gamma = 0.9

# Mixed precision
dtype = "float16"

run_name = "Task01_2d_cor"
