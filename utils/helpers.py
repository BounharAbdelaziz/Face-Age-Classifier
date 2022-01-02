import torch
from model.block import *
from torch import nn
from torch.nn import init

import numpy as np

def write_logs_tb(tb_writer_loss, tb_writer_img, example_img, age_pred, age_real, losses, step, epoch, hyperparams, with_print_logs=True):

    for k,v in losses.items():
        tb_writer_loss.add_scalar(
            k, v, global_step=step
        )
    
    age_pred = np.argmax(age_pred.cpu().numpy())

    tb_writer_loss.add_scalar(
        "age_pred", age_pred, global_step=step
    )

    tb_writer_loss.add_scalar(
        "age_real", age_real, global_step=step
    )

    tb_writer_img.add_image(
        "Example image", 
        example_img, global_step=step
    )

    if with_print_logs :
        print(f"Epoch [{epoch}/{hyperparams.n_epochs}]", sep=' ')
        for k,v in losses.items():
            print(f"{k} : [{v:.4f}]", sep=' - ', end=' - ')
            
    
def compute_nbr_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    if isinstance(net, nn.DataParallel):
        network_name = net.module.__class__.__name__
    else:
        network_name = net.__class__.__name__

    # print('[INFO] Initializing network %s with %s' % (network_name, init_type))
    net.apply(init_func)  # apply the initialization function <init_func>

    return net


def define_network(net, data_device, gpu_ids):
    print(f"[INFO] Setting up {len(gpu_ids)} GPU(s) for the networks...")
    print(f"[INFO] .... using GPU(s) device_ids : {gpu_ids} ...")

    if len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net) # multi-GPUs
    net.to(data_device)
   
    return net

def init_net(net, data_device,  gpu_ids=[], init_type='kaiming', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    
    # Put network on device (CPU or GPUs)
    net = define_network(net, data_device, gpu_ids)

    # Initialize network weights
    print(f"[INFO] Initializing the network weights with {init_type}")
    net = init_weights(net, init_type, init_gain=init_gain)

    return net 

"""
def define_network(net, data_device, device_ids):
    if len(device_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(data_device)
        net = torch.nn.DataParallel(net, device_ids, output_device=data_device)
    return net
"""