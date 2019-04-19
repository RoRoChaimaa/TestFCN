# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
from BagData import dataloader
import pdb
import numpy as np 
import time
import visdom
import numpy as np
import matplotlib.pyplot as plt
from FCN import VGGNet
from FCN import FCNs
from FCN import FCN8s
from FCN import FCN8s
from FCN import FCN16s
from FCN import FCN32s





if __name__ == "__main__":
    vis = visdom.Visdom()
    vgg_model = VGGNet(requires_grad=True)
    fcn_model = FCN32s(pretrained_net=vgg_model, n_class=2)
    fcn_model = fcn_model.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)
    #input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    #y = torch.autograd.Variable(torch.randn(batch_size, n_class, h, w), requires_grad=False)
    saving_index =0
    for epo in range(100):
        saving_index +=1
        index = 0
        epo_loss = 0
        start = time.time()
        print("dataloader size :", len(dataloader))
        for item in dataloader:
            index += 1
            start = time.time()
            input = item['A']
            y = item['B']
            input = torch.autograd.Variable(input)
            y = torch.autograd.Variable(y)

            input = input.cuda()
            y = y.cuda()

            optimizer.zero_grad()
            output = fcn_model(input)
            output = torch.sigmoid(output)
            loss = criterion(output, y)
            loss.backward()
            iter_loss = loss.item()
            epo_loss += iter_loss
            optimizer.step()


            if np.mod(index, 20) == 1:
                output_np = output.cpu().data.numpy().copy()
                output_np = np.argmin(output_np, axis=1)
                y_np = y.cpu().data.numpy().copy()
                y_np = np.argmin(y_np, axis=1)
                print('epoch {}, {}/{}, loss is {}'.format(epo, index, len(dataloader), iter_loss))
                vis.close()
                vis.images(output_np[:, None, :, :], opts=dict(title='pred')) 
                vis.images(y_np[:, None, :, :], opts=dict(title='label')) 
# =============================================================================
#             plt.subplot(1, 2, 1) 
#             plt.imshow(np.squeeze(y_np[0, :, :]), 'gray')
#             plt.subplot(1, 2, 2) 
#             plt.imshow(np.squeeze(output_np[0, :, :]), 'gray')
#             plt.pause(0.5)
# =============================================================================
        print('epoch loss = %f'%(epo_loss/len(dataloader)))
        
        if np.mod(saving_index, 5)==1:
            torch.save(fcn_model, 'checkpoints_FCN32/fcn_model_{}.pt'.format(epo))
            print('saving checkpoints/fcn_model_{}.pt'.format(epo))

