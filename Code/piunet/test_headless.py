# import utils and basic libraries
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils import gen_sub, bicubic
from losses import cpsnr, cssim
from skimage import io
from zipfile import ZipFile
import torch

PATH_DATASET = '../../Dataset/probav_data_synthetic_X_50_50/'

band = 'NIR' 
# band = 'RED'
mu = 7433.6436
sigma = 2353.0723
Nimages=9

# load ESA test set (no ground truth)
X_test = np.load(os.path.join(PATH_DATASET, f'X_{band}_val.npy'))

# print loaded dataset info
print('X_test: ', X_test.shape)

from config import Config
from model import PIUNET

MODEL_FILE = '../../Results/piunet/model_weights_best_20220316_2121_probav_data_synthetic_X_50_50_NIR.pt'

config = Config()

model = PIUNET(config)
model.cuda()

model.load_state_dict(torch.load(MODEL_FILE))


# create output directory
submission_time = time.strftime("%Y%m%d_%H%M")
SUBMISSION_DIR = '../../Results/piunet/'+'test_'+submission_time


if not os.path.exists(SUBMISSION_DIR):
    os.mkdir(SUBMISSION_DIR)

# vanilla
def predict_image(x_lr, dataset_mu, dataset_sigma, to_numpy=False):
    with torch.no_grad():
        model.eval()
        x_lr = torch.Tensor(np.transpose(x_lr,(0,3,1,2)).astype(np.float32)).to("cuda")
        x_sr, sigma_sr = model((x_lr-dataset_mu)/dataset_sigma)
        x_sr = x_sr*dataset_sigma + dataset_mu
        sigma_sr = torch.exp(sigma_sr + torch.log(torch.Tensor((dataset_sigma,)).to("cuda")))
    if to_numpy:
        return x_sr.permute(0,2,3,1).detach().cpu().numpy(), sigma_sr.permute(0,2,3,1).detach().cpu().numpy()
    else:
        return x_sr, sigma_sr


# rotational self-ensemble
def predict_image_se(x_lr, dataset_mu, dataset_sigma, to_numpy=True):
    with torch.no_grad():
        model.eval()
        for r in [0,1,2,3]:
            xr_lr = np.rot90(x_lr, k=r, axes=(1,2))
            xr_lr = torch.Tensor(np.transpose(xr_lr,(0,3,1,2)).astype(np.float32)).to("cuda")
            x_sr, sigma_sr = model((xr_lr-dataset_mu)/dataset_sigma)
            x_sr = x_sr*dataset_sigma + dataset_mu
            sigma_sr = torch.exp(sigma_sr + torch.log(torch.Tensor((dataset_sigma,)).to("cuda")))
            
            
            x_sr = x_sr.permute(0,2,3,1).detach().cpu().numpy()
            sigma_sr = sigma_sr.permute(0,2,3,1).detach().cpu().numpy()
            if r==0:
                x_sr_all = np.rot90(x_sr, k=-r, axes=(1,2))/4.0
                sigma_sr_all = np.rot90(sigma_sr, k=-r, axes=(1,2))/4.0
            else:
                x_sr_all = x_sr_all + np.rot90(x_sr, k=-r, axes=(1,2))/4.0
                sigma_sr_all = sigma_sr_all + np.rot90(sigma_sr, k=-r, axes=(1,2))/4.0
               
        return x_sr_all, sigma_sr_all
    

X_preds = []

X_test = X_test[...,:Nimages]
for index in tqdm(range(X_test.shape[0])):
    x_pred, sigma_pred = predict_image(X_test[index:index+1], mu, sigma, to_numpy=True)
    X_preds.append(x_pred)

def savePredictions(x, band, submission_dir):
    # Modified to work on single imgset in validation set => imgset0792
    """RAMS save util"""
    if band == 'NIR':
        i = "0596"
    elif band == 'RED':
        i = 1160
        
    for index in tqdm(range(len(x))):
        if i == "0596":
            io.imsave(os.path.join(submission_dir, f'imgset{i}.png'), x[index][0,:,:,0].astype(np.uint16),
                      check_contrast=False)
            break
        i+=1
        
savePredictions(X_preds, band, SUBMISSION_DIR)
