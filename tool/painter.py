import cv2
import skimage.transform

import numpy as np

def painter1():
    img_ori_1 = cv2.imread('./imgs/preprocess_2.jpg')
    img_ori_2 = cv2.imread('./imgs/preprocess_4.jpg')
    img_ori_3 = cv2.imread('./imgs/preprocess_6.jpg')
    img_ori_4 = cv2.imread('./imgs/preprocess_8.jpg')

    img_ori_3 = cv2.resize(img_ori_3, (496,279))
    img_ori_4 = cv2.resize(img_ori_4, (496,279))

    img_crop_1 = cv2.imread('./imgs/preprocess_1.jpg')
    img_crop_2 = cv2.imread('./imgs/preprocess_3.jpg')
    img_crop_3 = cv2.imread('./imgs/preprocess_5.jpg')
    img_crop_4 = cv2.imread('./imgs/preprocess_7.jpg')

    img_crop_1 = cv2.resize(img_crop_1, (368,368))
    img_crop_2 = cv2.resize(img_crop_2, (368,368))
    img_crop_3 = cv2.resize(img_crop_3, (368,368))
    img_crop_4 = cv2.resize(img_crop_4, (368,368))


    print(img_ori_1.shape, img_ori_2.shape, img_ori_3.shape, img_ori_4.shape)
    print(img_crop_1.shape, img_crop_2.shape, img_crop_3.shape, img_crop_4.shape)


    img_ori_cat = np.zeros((279, 496*4 + 30, 3))

    img_ori_cat[:, :496, :] = img_ori_1
    img_ori_cat[:, 496+10:496*2+10, :] = img_ori_2
    img_ori_cat[:, 496*2+20:496*3+20, :] = img_ori_3
    img_ori_cat[:, 496*3+30:496*4+30, :] = img_ori_4

    cv2.imwrite('./imgs/ori_cat.jpg', img_ori_cat)


    img_crop_cat = np.zeros((368, 368*4+30, 3))

    img_crop_cat[:, :368, :] = img_crop_1
    img_crop_cat[:, 368+10:368*2+10, :] = img_crop_2
    img_crop_cat[:, 368*2+20:368*3+20, :] = img_crop_3
    img_crop_cat[:, 368*3+30:368*4+30, :] = img_crop_4

    cv2.imwrite('./imgs/crop_cat.jpg', img_crop_cat)

def painter2():
    
    res_cat = np.zeros((368, 368*8+70, 3))
    for i in range(8):
        img = cv2.imread('./imgs/res_{}.jpg'.format(i+1))
        img = skimage.transform.resize(img, (368,368))
        img = np.uint8(img*255.)
        res_cat[:, 368*i+10*i:368*(i+1)+10*i, :] = img

    cv2.imwrite('./imgs/res_cat.jpg', res_cat)



if __name__ == "__main__":
    painter2()
