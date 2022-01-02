import torch 
import torch.nn as nn
from model.block import LinearLayer, ConvResidualBlock, LinearResidualBlock

class AgeClassifier(nn.Module):
  # -----------------------------------------------------------------------------#

  def __init__(   self, 
                  norm_type='bn2d', 
                  norm_before=True, 
                  activation='lk_relu', 
                  alpha_relu=0.15, 
                  use_bias=True,
                  min_features = 32, 
                  max_features=512,
                  n_inputs=3, 
                  n_output = 64, 
                  output_dim=1,               
                  down_steps=4, 
                  use_pad=True, 
                  kernel_size=3,
                  input_h=256,
                  input_w=256,
              ):
    """ The age classifier is in an encoder shape, we encode the features to a smaller space of features and do the decisions. """
    
    super(AgeClassifier, self).__init__()    
    
    # to do the cliping in the encoder and decoder
    features_cliping = lambda x : max(min_features, min(x, max_features))

    ##########################################
    #####             Encoder             ####
    ##########################################


    self.input_layer = []
    # input layer    
    self.input_layer.append(
      ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type='none', norm_before=norm_before, 
                        activation=activation, alpha_relu=alpha_relu)
    )
    print("------------- input layer -------------")

    print(f"n_inputs : {n_inputs}")
    print(f"n_output : {n_output}")

    print("------------- encoder -------------")
    self.encoder = []

    for i in range(down_steps-1):
      
      if i == 0 :
        n_inputs = n_output
        n_output = features_cliping(n_output * 2)

      print(f"i : {i}")
      print(f"n_inputs : {n_inputs}")
      print(f"n_output : {n_output}")
      print("---------------------------")
      self.encoder.append(
        ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu)
      )

      if i != down_steps-1 :
        n_inputs = features_cliping(n_inputs * 2)
        n_output = features_cliping(n_output * 2)

    self.input_layer = nn.Sequential(*self.input_layer)
    self.encoder = nn.Sequential(*self.encoder)

    self.flatten = nn.Flatten()
    
    new_h, new_w = input_h // (2**down_steps), input_w // (2**down_steps)
    flattened_dim = n_output*new_h*new_w
    
    self.residual_linears = []

    for i in range(4):
      self.residual_linears.append(
        LinearResidualBlock(  in_features=flattened_dim, out_features=flattened_dim, 
                              use_bias=use_bias, norm_type='bn1d', norm_before=norm_before, 
                              activation=activation, alpha_relu=alpha_relu))

    self.residual_linears = nn.Sequential(*self.residual_linears)
    # print("flattened_dim : ",flattened_dim)
    # print("n_output * 16 * 16 : ",n_output * 16 * 16)
    self.out_layer = LinearLayer(in_features=flattened_dim, out_features=output_dim, norm_type='none', 
                                activation='sigmoid', alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)

  # -----------------------------------------------------------------------------#

  def forward(self, x) :
    # print(f'AgeClassifier input : {x.shape}')
    out = self.input_layer(x)
    # print(f'input_layer output : {out.shape}')

    out = self.encoder(out)
    # print(f'encoder output : {out.shape}')
    out = self.flatten(out)    
    # print(f'flatten output : {out.shape}')
    out = self.residual_linears(out)
    # print(f'residual_linears output : {out.shape}')
    out = self.out_layer(out)
    # print(f'out_layer output : {out.shape}')

    return out
    
  # -----------------------------------------------------------------------------#
  