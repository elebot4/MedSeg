# 2D Coronal slice training - mobile optimized
# Training on coronal slices for different anatomical view

# Data settings  
input_shape = (256, 256)  # 2D coronal slices
batch_size = 8
slice_mode = 'cor'  # coronal slices

# Model architecture (mobile-friendly)
num_stages = 5  
base_chs = 32   
dropout = 0.1

# Training settings
nb_epochs = 500
learning_rate = 1e-3
weight_decay = 5e-3

# Mixed precision
dtype = 'float16'

run_name = "Task01_2d_cor"
