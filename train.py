import torch
import numpy as np

from data_preprocessing.data_generator import CACD2000Data
from torch.utils.data import DataLoader
from model.hyperparameters import Hyperparameters
from model.network import AgeClassifier
from model.age_clf import AgeClassifierNet

import utils.helpers as helper

from pathlib import Path

# fix seeds for reproducibility
__seed__ = 42
torch.manual_seed(__seed__)
np.random.seed(__seed__)


batch_size=64
experiment="AgeClassifier_adam_lr_expsch_5c"

path = Path.cwd() / "check_points" / experiment
try :
    path.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print(f'[INFO] Checkpoint directory already exists')
else:
    print(f'[INFO] Checkpoint directory has been created')

## Init Models
hyperparams = Hyperparameters(lambda_ce=1, n_epochs=100, lr=5e-5, batch_size=batch_size, n_output_clf=5, max_lr=0.1, show_advance=5)

# dumping hyperparams to keep track of them for later comparison/use
path_dump_hyperparams = path / "train_options.txt"
hyperparams.dump_values(path_dump_hyperparams)

## Init Dataloader
dataset = CACD2000Data(img_dir="../datasets/CACD2000/", h=256, w=256, device=hyperparams.device, n_class=hyperparams.n_output_clf)

## Init dataloarder
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)#, num_workers=hyperparams.num_threads) 

if batch_size > 1 :
    norm_type='bn2d'
else :
    norm_type='none'



age_clf = AgeClassifier(
                    norm_type='bn2d', 
                    norm_before=True, 
                    activation='lk_relu', 
                    alpha_relu=0.2, 
                    use_bias=True,
                    min_features = 16, 
                    max_features=512,
                    n_inputs=3, 
                    n_output = 32, 
                    output_dim=hyperparams.n_output_clf,               
                    down_steps=8, 
                    use_pad=True, 
                    kernel_size=3,
                    input_h=256,
                    input_w=256,
        )

print("-----------------------------------------------------")
print(f"[INFO] Number of trainable parameters for the AgeClassifier : {helper.compute_nbr_parameters(age_clf)}")
print("-----------------------------------------------------")
device_ids = [i for i in range(torch.cuda.device_count())]

if hyperparams.device != 'cpu':
    # using DataParallel tu copy the Tensors on all available GPUs
    device_ids = [i for i in range(torch.cuda.device_count())]    
    age_clf = helper.init_net( net=age_clf, 
                                    data_device=hyperparams.device,  gpu_ids=device_ids, 
                                    init_type=hyperparams.weights_init, init_gain=hyperparams.init_gain)
print("-----------------------------------------------------")


network = AgeClassifierNet(age_clf, hyperparams, experiment)

## Start Training

network.train(  dataloader=dataloader, 
                h=256, 
                w=256, 
                ckpt="./check_points/")

## Save trained models