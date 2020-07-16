import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

import os
import matplotlib.pyplot as plt
import numpy as np

from MPII import MPII
from model import CPM
from utils import AverageMeter
import img_utils as imgutils
from utils import save_checkpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ERROR+FATAL


cuda = torch.cuda.is_available()

train_losses = AverageMeter()
test_losses = AverageMeter()


def train(device, optimizer, model, criterion):
    
    model.train()
    train_dataset = MPII(is_training=True)
    train_loader = torchdata.DataLoader(train_dataset, batch_size=4, shuffle=True, \
                                        collate_fn=train_dataset.collate_fn, num_workers=4)

    for i, (img, heatmap, centermap) in enumerate(train_loader):

        img = torch.FloatTensor(img).to(device)
        heatmap = torch.FloatTensor(heatmap).to(device)
        centermap = torch.FloatTensor(centermap).to(device)

        score1, score2, score3, score4, score5, score6 = model(img, centermap)

        loss1 = criterion(heatmap, score1)
        loss2 = criterion(heatmap, score2)
        loss3 = criterion(heatmap, score3)
        loss4 = criterion(heatmap, score4)
        loss5 = criterion(heatmap, score5)
        loss6 = criterion(heatmap, score6)

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        train_losses.update(loss.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(device, model, criterion):

    model.eval()
    test_dataset = MPII(is_training=False)
    test_loader = torchdata.DataLoader(test_dataset, batch_size=1, shuffle=False, \
                                        collate_fn=test_dataset.collate_fn, num_workers=4)

    for i, (img, heatmap, centermap) in enumerate(test_loader):
        img = torch.FloatTensor(img).to(device)
        heatmap = torch.FloatTensor(heatmap).to(device)
        centermap = torch.FloatTensor(centermap).to(device)

        score1, score2, score3, score4, score5, score6 = model(img,centermap)

        loss1 = criterion(heatmap, score1)
        loss2 = criterion(heatmap, score2)
        loss3 = criterion(heatmap, score3)
        loss4 = criterion(heatmap, score4)
        loss5 = criterion(heatmap, score5)
        loss6 = criterion(heatmap, score6)

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        test_losses.update(loss.item(), img.size(0))

        # show predicted heatmaps 
        if i == 0:
            heatmaps_pred_copy = score6[0]
            heatmaps_copy = heatmap[0] 
            img_copy = img[0]
            heatmaps_pred_np = heatmaps_pred_copy.detach().cpu().numpy()
            heatmaps_pred_np = np.transpose(heatmaps_pred_np, (1, 2, 0))
            heatmaps_np = heatmaps_copy.detach().cpu().numpy()
            heatmaps_np = np.transpose(heatmaps_np, (1, 2, 0))
            img_np = img_copy.detach().cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            
            imgutils.show_heatmaps(img_np, heatmaps_pred_np, num_fig=1)
            imgutils.show_heatmaps(img_np, heatmaps_np, num_fig=2)
            plt.pause(0.5)

            

        
def main():
    plt.ion()
    if cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    model = CPM(k = 16).to(device)
    print('USE: ', device)
    epoches = 50   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss(reduction='mean')

    best_test_loss = 100.0
    print('==================== START TRAINING ====================')
    for e in range(epoches):
        train(device, optimizer, model, criterion)

        print('Epoch: {} || Training Loss: {}'.format(e+1, train_losses.avg))
        train_losses.reset()

        if (e+1)%1 == 0:
            test(device, model, criterion)
            print('Epoch: {} || Testing Loss: {}'.format(e+1, test_losses.avg))
            path_ckpt = './models/cpm_epoch_' + str(e+1)
            if test_losses.avg < best_test_loss:
                # save the model
                save_checkpoint(model.state_dict(), True, path_ckpt)
                print('===============CHECKPOINT PARAMS SAVED AT EPOCH %d ===============' %(e+1))
                best_test_loss = test_losses.avg
            else:
                save_checkpoint(model.state_dict(), False, path_ckpt)
                

            test_losses.reset()
    
    plt.ioff()
    print(' Training Complete !')



if  __name__ == "__main__":
    main()


