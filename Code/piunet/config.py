from enum import Enum

class Band(Enum):
    NIR = "NIR"
    RED = "RED"

class Config(object):

    def __init__(self):
        datadir = "Dataset/probav_data_synthetic1/"
        band =  Band.RED                             # Band.NIR | Band.RED


        self.model_name = datadir.split('/')[1] + '_' + band 

        self.train_lr_file    = datadir + "X_" + band + "_train.npy"
        self.train_hr_file    = datadir + "y_" + band + "_train.npy"
        self.train_masks_file = datadir + "y_" + band + "_train_masks.npy"
        self.val_lr_file      = datadir + "X_" + band + "_val.npy"
        self.val_hr_file      = datadir + "y_" + band + "_val.npy"
        self.val_masks_file   = datadir + "y_" + band + "_val_masks.npy"
        self.max_train_scenes = 393 if band == "NIR" else 415
        
        self.device = "cuda"
        self.validate = True

        # architecture
        self.N_feat = 42
        self.R_bneck = 8
        self.N_tefa = 16
        self.N_heads = 1 
        self.patch_size = 32

        # learning
        self.batch_size = 18
        self.N_epoch = 150 # ? maybe reduce?
        self.learning_rate = 1e-4
        self.workers = 5

        # logging
        self.log_every_iter = 100
        self.validate_every_iter = 1000
        self.save_every_iter = 1000
