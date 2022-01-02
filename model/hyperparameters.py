import torch

class Hyperparameters():
  
  def __init__( self, 
                lr=0.0003, 
                batch_size=32, 
                n_epochs=400, 
                norm_type='bn2d',
                norm_before=False, 
                down_steps=6, 
                min_features=32, 
                max_features=512, 
                n_output_clf=1, 
                n_ages_classes=5, 
                alpha_relu=0.2, 
                show_advance=5, 
                save_weights=5000,
                lambda_ce=150,
                num_threads=4,
                weight_decay=0.0005, 
                momentum=0.9,
                max_lr=0.1, 
                step_size_up=3, 
                mode="exp_range", 
                gamma=0.85,
                weights_init='kaiming',
                init_gain=0.02,
                validation_percentage=0.2,
              ):

    # Training device
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # optimization
    self.lr = lr
    self.weight_decay = weight_decay
    self.momentum = momentum

    # LR scheduler
    self.max_lr = max_lr
    self.step_size_up = step_size_up 
    self.mode = mode
    self.gamma = gamma

    self.weights_init = weights_init
    self.init_gain = init_gain

    # weights of losses
    self.lambda_ce = lambda_ce

    self.batch_size = batch_size
    self.n_epochs = n_epochs 

    # Network specifications
    self.norm_type = norm_type 
    self.norm_before = norm_before
    self.alpha_relu=alpha_relu
    self.min_features = min_features 
    self.max_features = max_features 
    self.down_steps = down_steps 
    self.n_output_clf = n_output_clf # 1 for regression or more for classification

    # Data loading
    self.num_threads = num_threads
    self.validation_percentage = validation_percentage

    # Logs
    self.show_advance=show_advance
    self.save_weights=save_weights
    

    # number of ages intervals (n_ages_classes=5 -> from 1 to 50, step of 10 ages)
    self.n_ages_classes = n_ages_classes

  def dump_values(self, path):

    with open(path, "w") as f:
      for attr, value in self.__dict__.items():
        f.writelines(attr + " : " + str(value) + "\n")
      f.close()