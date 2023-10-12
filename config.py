import os
import numpy
import torch
import random

# deterministic!
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

ROOT = os.path.realpath(os.path.join(os.path.abspath(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "stored_models")
MAX_PROCESSES = 8

# MAX_POST_TRAIN_TIMES = 5
MAX_POST_TRAIN_TIMES = 1    # best to set 3, set 1 to accelerate
MAX_TRAINING_THRESH = 300
MAX_COMBINATION_SIZE = 4
DEFAULT_XSI_THRESHOLD = 0.2
DEFAULT_VALID_THRESHOLD = 0 # 1e-5 # 1e-4
MAKE_COMBINATION = False
BASE_ADDITION_ON_PT = False    
# IMPORTANT: BASE_ADDITION_ON_PT can not be True, because, training base on pt will surely decrease the score  
CALCULATE_REAL_REL = False
MAX_EMBEDDING_DIFF_L2 = 0.1
NORMALIZE_DIFF = False
MAX_GROUP_CNT = 3
MAX_GROUP_SIZE = 100
MULTI_THREAD = 1



######
# To make the verify rsult stable, run multiple times
VERIFY_TIMES = 1
multi_thread = False