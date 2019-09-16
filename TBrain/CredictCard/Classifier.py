import torch.nn as nn
import math

"""
    Binary Classifier
"""

class Binary_Classifier(nn.Module):
    def __init__(self, in_dim):
        super(Binary_Classifier, self).__init__()
        self.classifier = nn.Sequential(
                nn.Linear(in_dim, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Dropout(0.4),
                
                nn.Linear(4096,2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048,2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Dropout(0.35),  
                
                nn.Linear(2048,1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(0.3),                  
                
                nn.Linear(1024,512),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Dropout(0.25),  
                
                nn.Linear(512,2) # Binary          
        )
#        # Initialize weights(L2 norm)
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.classifier(x)
        return out
