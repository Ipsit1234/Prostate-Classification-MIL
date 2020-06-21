#!/usr/bin/env python
# coding: utf-8

# In[35]:


import os
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms, models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
                
        self.attention = nn.Sequential(
        nn.Linear(self.L,self.D),
        nn.Tanh(),
        nn.Linear(self.D,self.K),
        )
        self.net = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        
        
        self.classifier = nn.Sequential(
        nn.Linear(self.L*self.K,128),
        nn.ReLU(),
        nn.Linear(128,2),
#         nn.Sigmoid(),
        nn.Softmax(dim=1)
        )
        
    def forward(self,x):
#         x = x.squeeze(0)
        H = self.net(x)
#         print(H.shape)
        H = H.view(-1,512)
        #print(H.shape)
#         x = x.squeeze(0)
#         H = self.feature_extractor_part1(x)
#         print(H.shape)
#        H = H.view(-1,100*118*118)
#         H = self.feature_extractor_part2(H) # NxL
#         print(H.shape)
        
        A = self.attention(H) # NxK
        A = torch.transpose(A,1,0) # KxN
        A = F.softmax(A,dim=1) # softmax over N
        
        M = torch.mm(A,H) # KxL
        
        Y_prob = self.classifier(M)
#         Y_hat = torch.ge(Y_prob,0.5).float()
        
        return Y_prob #, Y_hat, A


# In[36]:


net = Attention()


# In[11]:


class MIL_Data(torch.utils.data.Dataset):
    def __init__(self,bags_dir = 'bags',dataset_type='train',transform=None,labels_file='train_labels.csv'):
        super(MIL_Data,self).__init__()
        self.dataset_type = dataset_type
        self.bags_dir = bags_dir
        self.transform = transform
        self.labels = pd.read_csv(labels_file,index_col=0)
        self.train_files = []
        self.val_files = []
        for file in os.listdir(self.bags_dir):
            if file.startswith('ZT76'):
                self.val_files.append(file)
            else:
                self.train_files.append(file)
    def __len__(self):
        counts = 0
        for file in os.listdir(self.bags_dir):
            if file.startswith('ZT76'):
                counts += 1
        if self.dataset_type == 'train':
            return len(os.listdir(self.bags_dir)) - counts
        elif self.dataset_type == 'valid':
            return counts
        return len(os.listdir(self.bags_dir))
    def __getitem__(self,idx):
        if self.dataset_type == 'valid':
            bag_name = self.val_files[idx]
            true_label = torch.LongTensor(self.labels.loc[bag_name])
            bag_tensor = torch.zeros((len(os.listdir(os.path.join(self.bags_dir,bag_name))),3,310,310))
            if bag_name.startswith('ZT76'):
                for ind,image in enumerate(os.listdir(os.path.join(self.bags_dir,bag_name))):
                    img = cv2.imread(os.path.join(self.bags_dir,bag_name + '/' + image))
                    if self.transform:
                        img = self.transform(img)
                    bag_tensor[ind] = img
            return (bag_tensor,true_label)
        elif self.dataset_type == 'train':
            bag_name = self.train_files[idx]
            true_label = torch.LongTensor(self.labels.loc[bag_name])
            bag_tensor = torch.zeros((len(os.listdir(os.path.join(self.bags_dir,bag_name))),3,310,310))
            if not bag_name.startswith('ZT76'):
                for ind,image in enumerate(os.listdir(os.path.join(self.bags_dir,bag_name))):
                    img = cv2.imread(os.path.join(self.bags_dir,bag_name + '/' + image))
                    if self.transform:
                        img = self.transform(img)
                    bag_tensor[ind] = img
            return (bag_tensor,true_label)


# In[12]:


train_data = MIL_Data(dataset_type='train',transform=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))
train_loader = torch.utils.data.DataLoader(train_data,batch_size=1,num_workers=0)


# In[13]:


valid_data = MIL_Data(dataset_type='valid',transform=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))
valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=1,num_workers=0)


# In[14]:


torch.save(train_loader,'train_loader.pth')
torch.save(valid_loader,'val_loader.pth')



# In[16]:


net = net.cuda()
writer = SummaryWriter('runs/MIL')
optimizer = optim.Adam(net.parameters(),lr=1e-4)
# scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-7,max_lr=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.2)
loss_function = nn.CrossEntropyLoss()


# In[17]:


EPOCHS = 50
# transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
losst = 0.0
lossv = 0.0
for epoch in range(EPOCHS):
    losst = 0.0
    lossv = 0.0
    for i,batch in tqdm(enumerate(train_loader,0)):
        net.train()
        bag_tensor,target = batch
        optimizer.zero_grad()
        output = net(bag_tensor[0].cuda())
        
        loss = loss_function(output,target[0].cuda())
        loss.backward()
        optimizer.step()
        losst += loss.item()
    print("Train Loss: {}".format(losst/len(train_loader)))
    writer.add_scalar('Train Loss',losst/len(train_loader),epoch)
    for j,batch in tqdm(enumerate(valid_loader,0)):
        net.eval()
        with torch.no_grad():
            bag_tensor,target = batch
            if not bag_tensor[0].shape[0] == 1:
                output = net(bag_tensor[0].view(-1,3,155,155).cuda())
                loss = loss_function(output,target[0].cuda())
                lossv += loss.item()
        scheduler.step(lossv/len(valid_loader))
    print("Val Loss: {}".format(lossv/len(valid_loader)))
    writer.add_scalar('Val Loss',lossv/len(valid_loader),epoch)




# In[ ]:


checkpoint = {'model':Attention(),
             'state_dict':net.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler':scheduler.state_dict(),
             'epoch':50,
             }
torch.save(checkpoint, 'checkpoint.pth')



