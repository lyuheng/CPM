import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import skimage.transform

from model import CPM
import img_utils as imgutils
import utils


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ERROR+FATAL

MPII_FILE_DIR = '../mpii_human_pose_v1'

cuda = torch.cuda.is_available()


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
        self.test_list  = self.test_list[:200]

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
        
        height, width, _ = img_crop.shape
        img = np.transpose(img_crop, (2,0,1))/255.0
        return img

    def __len__(self):
        if self.is_training:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def collate_fn(self, batch):
        imgs = np.stack(batch, axis=0)
        return imgs



def test():
    """
    img: (H,W,3)
    """
    
    if cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'

    print(device)

    model = CPM(k = 16).to(device)

    MODEL_DIR = './models/cpm_epoch_43_best.pkl'
    model = CPM(k=16)
    model.load_state_dict(torch.load(MODEL_DIR))
    model = model.to(device)

    model.eval()
    mpii = MPII_(is_training=False)

    eval_loader = torchdata.DataLoader(mpii, batch_size=1, shuffle=True, num_workers=4, collate_fn=mpii.collate_fn)
    for i, imgs in enumerate(eval_loader):
        
        imgs_torch = torch.FloatTensor(imgs).to(device)
        # centermap is just the center of image, i.e. (184,184)
        centermap = imgutils.generate_heatmap(np.zeros((368,368)), pt=[184,184])
        centermap_torch = torch.FloatTensor(centermap).unsqueeze(0).unsqueeze(1).to(device) # add dim

        score1, score2, score3, score4, score5, score6 = model(imgs_torch, centermap_torch)

        score = {
            1: score1,
            2: score2,
            3: score3,
            4: score4,
            5: score5,
            6: score6,
        }

        joints = imgutils.heatmaps_to_coords(score6[0].cpu().detach().numpy().transpose((1,2,0))[:,:,:16] )

        img = imgs[0].transpose((1,2,0))
        img = np.uint8(255*img)
        cv2.imwrite('./imgs/ori_stages.jpg', img)


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

        for idx, name in dict_name.items():
            if idx == 0:
                continue
            paint(idx-1, name, img, score)


        #im = cv2.addWeighted(img, 0.4, score6, 0.6, 0)

        #cv2.imshow('head',im )

        #imgutils.show_stack_joints(imgs[0].transpose((1,2,0)), joints, draw_lines=True)

        if i == 0:
            break


def paint(num, name, img, score):
    img_cat = np.zeros((368, 368*6 + 50, 3))
    for e in range(6):
        #plt.subplot(2,3,e+1)
        s = score[e+1]
        s = s.cpu().detach().numpy()[0, num]
        s /= np.max(s)
        s = cv2.resize(s, (368,368))
        s = np.uint8(255*s)
        s = cv2.applyColorMap(s, cv2.COLORMAP_JET)
        im = 0.26*s + 0.74*img
        im = np.uint8(im)
        img_cat[:, 368*e+10*e:368*(e+1)+10*e, :] = im
        #plt.imshow(im)
    img_cat = img_cat.astype(np.uint8)
    cv2.imwrite('./imgs/{}_stages.jpg'.format(name), img_cat)
    
    

if __name__ == "__main__":
    test()


    