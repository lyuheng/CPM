import torch
import torch.utils.data as torchdata

import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt
import skimage

import img_utils as imgutils
import transform

MPII_FILE_DIR = '../mpii_human_pose_v1'


class MPII(torchdata.Dataset):
    """
    Dataset for MPII
    """
    def __init__(self, is_training=True, use_rotation=True, use_flip=True):
        """
        rotation not implemented
        """
        super(MPII, self).__init__()
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

        """
        dataset :  MPI
        isValidation :  1.0
        img_paths :  005808361.jpg
        img_width :  1280.0
        img_height :  720.0
        objpos :  [966.0, 340.0]
        joint_self :  [[804.0, 711.0, 1.0], [816.0, 510.0, 1.0], [908.0, 438.0, 1.0], [1040.0, 454.0, 1.0], [906.0, 528.0, 1.0], [883.0, 707.0, 1.0], [974.0, 446.0, 1.0], [985.0, 253.0, 1.0], [982.759, 235.969, 1.0], [962.241, 80.031, 1.0], [869.0, 214.0, 1.0], [798.0, 340.0, 1.0], [902.0, 253.0, 1.0], [1067.0, 253.0, 1.0], [1167.0, 353.0, 1.0], [1142.0, 478.0, 1.0]]
        scale_provided :  4.718
        joint_others :  [[667.0, 633.0, 1.0], [675.0, 462.0, 1.0], [567.0, 519.0, 1.0], [375.0, 504.0, 1.0], [543.0, 476.0, 0.0], [532.0, 651.0, 0.0], [471.0, 512.0, 1.0], [463.0, 268.0, 1.0], [472.466, 220.857, 1.0], [503.534, 66.143, 1.0], [702.0, 267.0, 1.0], [721.0, 386.0, 1.0], [584.0, 256.0, 1.0], [341.0, 280.0, 1.0], [310.0, 432.0, 1.0], [372.0, 496.0, 1.0]]
        scale_provided_other :  4.734
        objpos_other :  [489.0, 383.0]
        annolist_index :  7.0
        people_index :  1.0
        numOtherPeople :  1.0
        """

        # generate crop image
        #print(img)
        img_crop, pts_crop, cen_crop = imgutils.crop(img, ele_anno)
        pts_crop = np.array(pts_crop)
        cen_crop = np.array(cen_crop)
        
        height, width, _ = img_crop.shape
        train_img = np.transpose(img_crop, (2,0,1))/255.0

        train_heatmaps = imgutils.generate_heatmaps(np.zeros((46,46)), pts_crop/8)
        train_heatmaps = np.transpose(train_heatmaps, (2,0,1))

        train_centermap = imgutils.generate_heatmap(np.zeros((368,368)), cen_crop)
        train_centermap = np.expand_dims(train_centermap, axis=0)

        #return img_crop, pts_crop, cen_crop, img, pts, cen

        return train_img, train_heatmaps, train_centermap

    def __len__(self):
        if self.is_training:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def collate_fn(self, batch):

        imgs, heatmaps, centermap = list(zip(*batch))

        imgs = np.stack(imgs, axis=0)
        heatmaps = np.stack(heatmaps, axis=0)
        centermap = np.stack(centermap, axis=0)

        return imgs, heatmaps, centermap

        # imgs_crop, pts_crop, cens_crop, imgs, pts, cens = list(zip(*batch))

        # return imgs_crop, pts_crop, cens_crop, imgs, pts, cens
    

# Test preprocessing
def main():
    #plt.ion()
    mpii = MPII(is_training=False)
    dataloader = torchdata.DataLoader(mpii, batch_size=1, shuffle=True, collate_fn=mpii.collate_fn)
    for i, (img, heatmap, centermap) in enumerate(dataloader):
        
        #print(img.shape, heatmap.shape, centermap.shape)
        #print(img_crop[0].shape)

        #imgutils.show_stack_joints(img_crop[0], pts_crop[0], cen_crop[0], num_fig=2*i+1)
        #imgutils.show_stack_joints(img[0], pts[0], cen[0], num_fig=2*i+2)
        imgutils.show_heatmaps(img[0].transpose(1,2,0), heatmap[0].transpose(1,2,0))
        #plt.pause(5)
        if i == 0:
            break
    #plt.ioff()

if __name__ == "__main__":
    main()
    #pass