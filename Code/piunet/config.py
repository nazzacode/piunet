class Config(object):

    def __init__(self):

        datadir = "Dataset/probav_data_synthetic1/"

        # NIR 
        self.train_lr_file = datadir + "X_NIR_train.npy"
        self.train_hr_file = datadir + "y_NIR_train.npy"
        self.train_masks_file = datadir + "y_NIR_train_masks.npy"
        self.val_lr_file = datadir + "X_NIR_val.npy"
        self.val_hr_file = datadir + "y_NIR_val.npy"
        self.val_masks_file = datadir + "y_NIR_val_masks.npy"
        self.max_train_scenes = 393
        # RED
        #self.train_lr_file = "Dataset/X_RED_train.npy"
        #self.train_hr_file = "Dataset/y_RED_train.npy"
        #self.train_masks_file = "Dataset/y_RED_train_masks.npy"
        #self.val_lr_file = "Dataset/X_RED_val.npy"
        #self.val_hr_file = "Dataset/y_RED_val.npy"
        #self.val_masks_file = "Dataset/y_RED_val_masks.npy"
        #self.max_train_scenes = 415
        
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
        self.N_epoch = 750 # ? maybe reduce?
        self.learning_rate = 1e-4
        self.workers = 5

        # logging
        self.log_every_iter = 100
        self.validate_every_iter = 1000
        self.save_every_iter = 1000
