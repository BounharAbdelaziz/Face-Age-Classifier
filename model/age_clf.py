from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torchvision
from torch.optim import SGD, Adam, AdamW

import utils.helpers as helper
from model.loss import CELoss

import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR

class AgeClassifierNet():
  
  def __init__(self, age_clf, hyperparams, experiment="AgeClassifierNet_v1"):

    self.age_clf = age_clf
    self.hyperparams = hyperparams

    # we use SGD optimizer for both Generator and Discriminator
    # self.opt_age_clf = SGD( self.age_clf.parameters(), 
    #                         lr=self.hyperparams.lr, 
    #                         weight_decay=self.hyperparams.weight_decay, 
    #                         momentum=self.hyperparams.momentum)

    self.opt_age_clf = Adam( self.age_clf.parameters(), 
                            lr=self.hyperparams.lr)#, 
                            # weight_decay=self.hyperparams.weight_decay)

    # self.scheduler_age_clf = CyclicLR(  self.opt_age_clf, 
    #                                     base_lr=self.hyperparams.lr, 
    #                                     max_lr=self.hyperparams.max_lr, 
    #                                     step_size_up=self.hyperparams.step_size_up, 
    #                                     mode=self.hyperparams.mode, 
    #                                     gamma=self.hyperparams.gamma)

    self.scheduler_age_clf = ExponentialLR(self.opt_age_clf, gamma=self.hyperparams.gamma)

    # Loss functions
    self.loss_CE = CELoss(device=self.hyperparams.device)
    self.eps = 1e-8

    # Hyper-parameters
    self.lambda_ce = hyperparams.lambda_ce # softmax loss

    # Tensorboard logs
    self.experiment = experiment
    self.tb_writer_img = SummaryWriter(f"logs/{self.experiment}/image_{self.experiment}")
    self.tb_writer_loss = SummaryWriter(f"logs/{self.experiment}/loss_train_{self.experiment}")

    
  def backward_age_clf(self, label, prediction):
    loss_CE =  self.lambda_ce*self.loss_CE(label, prediction) + self.eps

    with torch.autograd.set_detect_anomaly(True) : # set to True only during debug
      loss_CE.backward()
    return loss_CE 

  def train(self, dataloader, h=256, w=256, ckpt="./check_points/", with_print_logs=True):

    step = 0
    self.PATH_CKPT = ckpt+self.experiment+"/"
    
    print("[INFO] Started training using device ",self.hyperparams.device,"...")

    
    for epoch in tqdm(range(self.hyperparams.n_epochs)):

      print("epoch = ",epoch," --------------------------------------------------------\n")
      
      for batch_idx, data in enumerate(dataloader) : 

        img_input, labels = data
        # print(f'img_input device {img_input.get_device()}')

        with torch.autograd.set_detect_anomaly(True) :

          outputs = self.age_clf(img_input)

          self.opt_age_clf.zero_grad()
          loss_CE = self.backward_age_clf(outputs, labels)
          self.opt_age_clf.step()

        # Logging advances
        if step % self.hyperparams.show_advance == 0 and step!=0:

          with torch.no_grad():
            
            age_pred = outputs[0]
            age_real = labels[0]

            example_img = img_input[0].reshape(1, 3, h, w)

            example_img = torchvision.utils.make_grid(example_img, normalize=True)

            losses = {}
            losses["loss_CE"] = loss_CE
            
            # lr schedulers
            losses["lr_age_clf"] = self.scheduler_age_clf.get_last_lr()[0]
            
            helper.write_logs_tb( self.tb_writer_loss, self.tb_writer_img,
                                  example_img, age_pred, age_real, losses, step, epoch, 
                                  self.hyperparams, with_print_logs=with_print_logs)


        if step % self.hyperparams.save_weights == 0 and batch_idx!=0 :

          # Saving weights advance
          print("[INFO] Saving weights...")
          torch.save(self.age_clf.state_dict(), os.path.join(self.PATH_CKPT,"Net_it_"+str(step)+".pth"))
        
        step = step + 1

      self.scheduler_age_clf.step()

    print("[INFO] Saving weights last step...")
    torch.save(self.age_clf.state_dict(), os.path.join(self.PATH_CKPT,"Net_last_it_"+str(step)+".pth"))
    print(f'Latest networks saved in : {self.PATH_CKPT}')
