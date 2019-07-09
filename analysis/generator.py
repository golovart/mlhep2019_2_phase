import torch
import torch.nn as nn
import torch.nn.functional as F
NOISE_DIM = 10

def swish(x):
    return x * torch.sigmoid(x)

class ModelGConvTranspose(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelGConvTranspose, self).__init__()
        self.fc1 = nn.Linear(self.z_dim + 2 + 3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.bn_l2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)
        self.bn_l3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 20736)
        
        self.dropout = nn.Dropout(p=0.3)
        
        self.conv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=1)
        nn.init.xavier_normal_(self.conv1.weight)
        self.conv2 = nn.ConvTranspose2d(128, 64, 3)
        nn.init.xavier_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, 3)
        nn.init.xavier_normal_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 16, 3)
        nn.init.xavier_normal_(self.conv4.weight)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.ConvTranspose2d(16, 8, 3)
        nn.init.xavier_normal_(self.conv5.weight)
        self.conv6 = nn.ConvTranspose2d(8, 1, 3)
        nn.init.xavier_normal_(self.conv6.weight)
        
        
    def forward(self, z, ParticleMomentum_ParticlePoint):
        #x = F.leaky_relu(self.fc1(
        #    torch.cat([z, ParticleMomentum_ParticlePoint], dim=1)
        #))
        #x = F.leaky_relu(self.fc2(x))
        #x = F.leaky_relu(self.fc3(x))
        #x = F.leaky_relu(self.fc4(x))
        x = swish(self.fc1(
            torch.cat([z, ParticleMomentum_ParticlePoint], dim=1)
        ))
        x = swish(self.bn_l2(self.fc2(x)))
        #x = self.dropout(swish(self.bn_l3(self.fc3(x))))
        x = swish(self.bn_l3(self.fc3(x)))
        x = swish(self.fc4(x))
        
        EnergyDeposit = x.view(-1, 256, 9, 9)
        
        #EnergyDeposit = F.leaky_relu(self.conv1(EnergyDeposit))
        #EnergyDeposit = F.leaky_relu(self.conv2(EnergyDeposit))
        #EnergyDeposit = F.leaky_relu(self.conv3(EnergyDeposit))
        #EnergyDeposit = F.leaky_relu(self.conv4(EnergyDeposit))
        #EnergyDeposit = F.leaky_relu(self.conv5(EnergyDeposit))
        EnergyDeposit = swish(self.conv1(EnergyDeposit))
        #EnergyDeposit = self.dropout(swish(self.bn2(self.conv2(EnergyDeposit))))
        EnergyDeposit = swish(self.bn2(self.conv2(EnergyDeposit)))
        EnergyDeposit = swish(self.bn3(self.conv3(EnergyDeposit)))
        #EnergyDeposit = self.dropout(swish(self.bn4(self.conv4(EnergyDeposit))))
        EnergyDeposit = swish(self.bn4(self.conv4(EnergyDeposit)))
        EnergyDeposit = swish(self.conv5(EnergyDeposit))
        EnergyDeposit = self.conv6(EnergyDeposit)

        return EnergyDeposit