import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, device='cuda'):
        super(CELoss, self).__init__()
        self.eps = 1e-8
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x, y):
        
        # x = x.squeeze()
        y = y.squeeze(1)
        # print(f'loss x.shape : {x.shape}')
        # print(f'loss y.shape : {y.shape}')
        # print(f'loss y : {y}')
         
        loss = self.criterion(x,y) + self.eps
        # print(f'loss  : {loss}')

        return loss

"""
class AgeLoss(nn.Module):
    def __init__(self, n_ages_classes=5, device='cuda'):
        super(AgeLoss, self).__init__()
        self.age_clf = AgeClassifier(n_ages_classes, device=device)
        # Includes a sigmoid layer before using the BCE Loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x, age_class):
        
        x_age = self.age_clf(x)   

        # print(F.sigmoid(x_age))
        # print(F.softmax(x_age))
        # print(age_class)
        # print("--------------------------")
        return sum([ self.criterion(x_age[i], age_class[i, :]) for i in range(x.shape[0]) ])
"""