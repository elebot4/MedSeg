# 2D axial slice training - lightweight baseline
# Fast training on axial slices for deployment-oriented experiments

# Data settings
input_shape = (256, 256)  # 2D axial slices
batch_size = 8
slice_mode = "axi"  # axial slices

# Model architecture (smaller footprint)
num_stages = 5  # Fewer stages for faster iteration
base_chs = 32  # Smaller base channels
dropout = 0.1

# Training settings
nb_epochs = 1000
learning_rate = 1e-2
weight_decay = 3e-5
optimizer = "SGD"
momentum = 0.99
scheduler = "PolyLR"
gamma = 0.9

# Mixed precision for efficiency
dtype = "float16"

#
run_name = "Task01_2d_axi"
