import torch
import torch.nn as nn

### 
args_discrete = 'contp','etymd','mcc','ecfg','insfg','stocn','stscd',\
            'ovrlt','flbmk','hcefg','csmcu','flg_3dsmk'


###


class My_AutoEncoder(nn.Module):
    def __init__(self, in_dim):
        super(My_AutoEncoder, self).__init__()           
        self.embedding = nn.Embedding()

        self.encoder = nn.Sequential(
                nn.Linear(in_dim,512),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                
                nn.Linear(512,256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(0.20),  
                
                nn.Linear(256,128),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                
                nn.Linear(128,64),
                nn.BatchNorm1d(64),
                nn.ReLU(True),                
                nn.Dropout(0.15),
                
                nn.Linear(64,32),
                nn.BatchNorm1d(32),
                nn.ReLU(True),

                nn.Linear(32,8),      
                )
        self.decoder = nn.Sequential(
                nn.Linear(8,32),
                
                nn.ReLU(True),
                nn.BatchNorm1d(32),
                nn.Linear(32,64),
                
                nn.Dropout(0.15),
                nn.ReLU(True),
                nn.BatchNorm1d(64),
                nn.Linear(64,128),
                
                nn.ReLU(True),
                nn.BatchNorm1d(128),
                nn.Linear(128,256),
                
                nn.Dropout(0.20),
                nn.ReLU(True),
                nn.BatchNorm1d(256), 
                nn.Linear(256,512),
                
                nn.ReLU(True),
                nn.BatchNorm1d(512),
                nn.Linear(512,in_dim),
                nn.Tanh()
                )
        self.confidence = nn.Linear(in_dim, 1)
    
    def forward(self, X_discrete, X_continously):
        
        
        code = self.encoder(x)
        out = self.decoder(code)
        confidence = self.confidence(out)
        return out, confidence
