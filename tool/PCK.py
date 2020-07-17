# coding: utf-8
import torch
import torch.utils.data as torchdata

import os
import json
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/lyuheng/vision/CPM_MPII')

from model import CPM
import img_utils as imgutils
import utils as utils



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ERROR+FATAL

MPII_FILE_DIR = '../mpii_human_pose_v1'

cuda = torch.cuda.is_available()

dict_name = {
            0: 'origin img',
            1: 'left ankle',
            2: 'left knee',
            3: 'left hip',
            4: 'right hip',
            5: 'right knee',
            6: 'right ankle',
            7: 'belly',
            8: 'chest',
            9: 'neck',
            10: 'head',
            11: 'left wrist',
            12: 'left elbow',
            13: 'left shoulder',
            14: 'right shoulder',
            15: 'right elbow',
            16: 'right wrist',
            17: 'background'
        }


class MPII_(torchdata.Dataset):
    """
    Dataset for MPII
    """
    def __init__(self, is_training=True, use_rotation=False, use_flip=False):
        """
        rotation not implemented. done.
        """
        super(MPII_, self).__init__()
        self.is_training = is_training
        self.use_rotation = use_rotation
        self.use_flip = use_flip
        json_dir = os.path.join(MPII_FILE_DIR, 'mpii_annotations.json')

        with open(json_dir) as f:
            self.anno = json.load(f)

        self.train_list = []  # 22246
        self.test_list =  []  # 2958

        for i, anno in enumerate(self.anno):
            if anno['isValidation'] == True:
                self.test_list.append(i)
            else:
                self.train_list.append(i)
        
        ### only use a sample to train  ###
        self.train_list = self.train_list[:2000]
        # self.test_list  = self.test_list[:200]  # use whole test set

    def __getitem__(self, index):

        if self.is_training:
            ele_anno = self.anno[self.train_list[index]]
        else:
            ele_anno = self.anno[self.test_list[index]]

        img_sz = (368,368)
        img_folder_dir = os.path.join(MPII_FILE_DIR, 'images')
        img_dir = os.path.join(img_folder_dir, ele_anno['img_paths'])
        img = imgutils.load_img(img_dir)
    
        pts = ele_anno['joint_self']
        cen = ele_anno['objpos'].copy()
        scale = ele_anno['scale_provided']

        # generate crop image
        #print(img)
        img_crop, pts_crop, cen_crop = imgutils.crop(img, ele_anno, self.use_rotation, self.use_flip)
        pts_crop = np.array(pts_crop)
        cen_crop = np.array(cen_crop)

        img = np.transpose(img_crop, (2,0,1))/255.0
        return img, pts_crop[:, :2]

    def __len__(self):
        if self.is_training:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def collate_fn(self, batch):

        imgs, pts = list(zip(*batch))
        imgs = np.stack(imgs, axis=0)
        return imgs, pts




def visualize_accuracy(a, start=0.01, end=0.5, showlist=None, resolution=100):
    """ Compute the Accuracy in rate
    Args:
        a:              distance matrix like err in compute_distance 
        showlist:       list of joints to be shown
        resolutional:   determine the 
    """
    if showlist is None:
        showlist = range(0, 16)
    else:
        showlist = showlist

    plt.title("CPM PCKh benchmark on MPII")
    plt.xlabel("Normalized distance")
    plt.ylabel("Accuracy")

    dists = np.linspace(start, end, resolution)
    re = np.zeros((dists.shape[0], a.shape[1]), np.float)
    av = np.zeros((dists.shape[0]), np.float)
    for ridx in range(dists.shape[0]):
        print('[*]\tProcessing Result in normalized distance of', dists[ridx])
        for j in range(a.shape[1]):
            condition = a[:,j] <= dists[ridx]
            re[ridx, j] = len(np.extract(condition, a[:,j]))/float(a.shape[0])
            print('[*]\t", Global.joint_list[j], " Accuracy :\t\t', re[ridx, j])
        av[ridx] = np.average(re[ridx])
        print('[*]\t\tTOTAL Accuracy :\t\t', av[ridx])

    for j in showlist:
        plt.plot(dists, re[:, j], label=dict_name[j+1], linewidth = 2.0)
    plt.plot(dists, av, label="Average",linewidth = 3.0)
    plt.legend(loc='best')
    plt.grid(ls='--')
    # plt.show()

    plt.savefig('./imgs/PCKh.jpg', bbox_inches='tight', pad_inches=.1)


def compute_distance(model, metric='PCKh', debug=False):
    """ 
    Args:
        model:      model to load
        dataset:    dataset to generate image & ground truth
    Return:
        An normalized distance matrix with shape of num_of_img x joint_num
    """
    if metric == 'PCKh':   # distance between head and neck
        normJ_a = 9
        normJ_b = 8
    elif metric == 'PCK':
        normJ_a = 12
        normJ_b = 3
    else:
        raise ValueError

    test_dataset = MPII_(is_training=False)
    test_num = test_dataset.__len__()
    err = np.zeros((test_num, 16), dtype=np.float)

    paral = 2
    if debug:
        paral = 2

    test_loader = torchdata.DataLoader(test_dataset, batch_size=paral, collate_fn=test_dataset.collate_fn)

    for _iter, (img, pts) in enumerate(test_loader):
        
        paral = img.shape[0] # batch_sz

        pts = np.array(pts) # (batchsz,16,2)

        pts[np.where(pts == 0)] = -1  # set 0 points to -1 for PCK, a
        # im_list = []
        # j_list = []
        w_list = []

        for n in range(paral):
            w = (pts[n][:,1] != 0).astype(np.int)
            w_list.append(w)
        
        w = np.array(w_list) # (batchsz, 16)

        #   estimate by networkn
        img_torch = torch.FloatTensor(img).to(device)
        centermap = imgutils.generate_heatmap(np.zeros((368,368)), pt=[184,184])
        centermap_torch = torch.FloatTensor(centermap).unsqueeze(0).unsqueeze(1).to(device) # add dim
        centermap_torch = centermap_torch.repeat(paral, 1, 1, 1)

        _, _, _, _, _,hm_dt = model(img_torch, centermap_torch)

        hm_dt = hm_dt.permute(0,2,3,1)
        j_dt, w_dt = imgutils.heatmaps_to_weights_coords(hm_dt.cpu().detach().numpy()) # (batchsz, 16,2 in w,h) (batchsz, 16)

        w = np.transpose(np.hstack((np.expand_dims(w,1), np.expand_dims(w,1))), axes=[0,2,1]) # (bs,16,2)
        # (bs,16,2)
        w_dt = np.transpose(np.hstack((np.expand_dims(w_dt,1), np.expand_dims(w_dt,1))), axes=[0,2,1])
        for n in range(j_dt.shape[0]):
            temp_err = np.linalg.norm(w[n]*w_dt[n]*(pts[n,:,:]-j_dt[n,:,:]),axis=1) / (1e-6+np.linalg.norm(pts[n,normJ_a,:]-pts[n,normJ_b,:], axis=0))
            err[_iter*paral+n] = temp_err
        print('[*]\tTemp Error is ', np.average(err[_iter*paral:_iter*paral+paral], axis=0) )
        if debug:
            return err

    aver_err = np.average(err)
    print('[*]\tAverage PCKh Normalised distance is ', aver_err)

    return err

def get_err(debug=False):

    in_size = 368
    model = CPM(k=16)
    MODEL_DIR = './models/cpm_epoch_43_best.pkl'
    model = CPM(k=16)
    model.load_state_dict(torch.load(MODEL_DIR))
    model = model.to(device)
    model.eval()

    dist = compute_distance(model, metric='PCKh', debug=debug)
    return dist



if __name__ == '__main__':

    if cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    #np.save('./val_res.npy', get_err(debug=False))
    visualize_accuracy(np.load('./val_res.npy'), start=0.01, end=0.8, showlist=[0,1,4,5,10,11,14,15])