"""
Hingle Loss is used when performing SVM for multi-class classification
"""

import torch
import torch.nn as nn

class HingleLoss(nn.Module):

    def __init__(self, delta=1.0):
        super(HingleLoss, self).__init__()
        self.hl = nn.MultiMarginLoss(margin=delta)


    def forward(self, input, target):
        return self.hl(input, target)