import torch.nn as nn
import math

'''
    Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
    dilation=1, groups=1, bias=True, padding_mode='zeros') 
        in_channels(int) - Number of channels in the input image (1:binary , 3: RGB)
        out_channels(int) - Number of channels produced by the convolution,
                            i.e., the number of feature filter(neuron) will be conneted.
        
    BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    LeakyReLU(negative_slope=0.01, inplace=False)
    
    MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    Dropout(p=0.5, inplace=False)
'''

class VGG16(nn.Module):
    def __init__(self, in_dim, n_class):
        super(VGG16, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), # BatchNorm before activation
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),           
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), # BatchNorm before activation
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),            
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), # BatchNorm before activation
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),        
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), # BatchNorm before activation
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),       
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), # BatchNorm before activation
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),       
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)             
        )

        self.fc = nn.Sequential(
            nn.Linear(512*1*1, 4096), # need to check Linear input size
            nn.ReLU(True),
            nn.Dropout(), 
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(), 
            nn.Linear(1000, n_class)
        )
        # Initialize weights(L2 norm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # Flatten
        return self.fc(out)