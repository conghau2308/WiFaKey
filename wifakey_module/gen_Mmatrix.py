import numpy as np
from scipy.stats import ortho_group
import os

data_dir = './wifakey_module/data'
os.makedirs(data_dir, exist_ok=True)

M = ortho_group.rvs(dim=512) 
np.save(os.path.join(data_dir, 'M_matrix.npy'), M)
print(f"✅ Created {data_dir}/M_matrix.npy")