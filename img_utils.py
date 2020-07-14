import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage.transform
import skimage.filters


def load_img(dir):
    img = mpimg.imread(dir)
    return img


def generate_heatmap(heatmap, pt, sigma_valu=7):
    '''
    :param heatmap: should be a np zeros array with shape (H,W) (only 1 channel), not (H,W,1)
    :param pt: point coords, np array
    :param sigma: should be a tuple with odd values (obsolete)
    :param sigma_valu: value for gaussian blur
    :return: a np array of one joint heatmap with shape (H,W)
    This function is obsolete, use 'generate_heatmaps()' instead.
    '''
    heatmap[int(pt[1])][int(pt[0])] = 1
    heatmap = skimage.filters.gaussian(heatmap, sigma=sigma_valu)
    am = np.max(heatmap)
    heatmap = heatmap/am
    return heatmap



def generate_heatmaps(img, pts, sigma_valu=7):
    '''
    Generate 16 heatmaps
    :param img: np arrray img, (H,W,C)
    :param pts: joint points coords, np array, same resolu as img
    :param sigma: should be a tuple with odd values (obsolete)
    :param sigma_valu: vaalue for gaussian blur
    :return: np array heatmaps, (H,W,num_pts)
    '''
    H, W = img.shape[0], img.shape[1]
    num_pts = pts.shape[0]
    heatmaps = np.zeros((H, W, num_pts + 1))
    for i, pt in enumerate(pts):
        # Filter unavailable heatmaps
        if pt[0] == 0 and pt[1] == 0:
            continue
        # Filter some points out of the image
        if pt[0] >= W:
            pt[0] = W-1
        if pt[1] >= H:
            pt[1] = H-1
        heatmap = heatmaps[:, :, i]
        heatmap[int(pt[1])][int(pt[0])] = 1  # reverse sequence
        heatmap = skimage.filters.gaussian(heatmap, sigma=sigma_valu)  ##(H,W,1) -> (H,W)
        am = np.max(heatmap)
        heatmap = heatmap / am  # scale to [0,1]
        heatmaps[:, :, i] = heatmap

    heatmaps[:, :, num_pts] = 1.0 - np.max(heatmaps[:, :, :num_pts], axis=2)

    return heatmaps



def crop(img, ele_anno, use_scale=True, use_flip=True):

    H, W = img.shape[0], img.shape[1]
    s = ele_anno['scale_provided']      # like 4.431
    c = ele_anno['objpos']              # like [183.0, 327.0] center

    # Adjust center and scale
    if c[0] != -1:
        c[1] = c[1] + 15 * s
        s = s * 1.25
    ary_pts = np.array(ele_anno['joint_self'])  # (16, 3)
    
    ### select rows that are not [0,0,0] ###
    ary_pts_temp = ary_pts[np.any(ary_pts != [0, 0, 0], axis=1)]

    if use_scale:
        scale_rand = np.random.uniform(low=1.0, high=3.0)
    else:   
        scale_rand = 1

    # set offset in order not to crop key point
    W_min = max(np.min(ary_pts_temp, axis=0)[0] - s * 15 * scale_rand, 0) #2 
    H_min = max(np.min(ary_pts_temp, axis=0)[1] - s * 15 * scale_rand, 0) #2
    W_max = min(np.max(ary_pts_temp, axis=0)[0] + s * 15 * scale_rand, W) #8
    H_max = min(np.max(ary_pts_temp, axis=0)[1] + s * 15 * scale_rand, H) #7
    W_len = W_max - W_min  # 6
    H_len = H_max - H_min  # 5
    window_len = max(H_len, W_len)  # 6
    pad_updown = (window_len - H_len)/2 #.5
    pad_leftright = (window_len - W_len)/2 # 0

    # Calculate 4 corner position
    W_low = max((W_min - pad_leftright), 0) # 2
    W_high = min((W_max + pad_leftright), W) # 8
    H_low = max((H_min - pad_updown), 0)  # 1.5
    H_high = min((H_max + pad_updown), H) # 7.5

    # Update joint points and center
    """
    np.where(condition, x, y)
    """
    ary_pts_crop = np.where(ary_pts == [0, 0, 0], ary_pts, ary_pts - np.array([W_low, H_low, 0]))
    c_crop = c - np.array([W_low, H_low])

    img_crop = img[int(H_low):int(H_high), int(W_low):int(W_high), :].copy()

    # Pad when H, W different
    H_new, W_new = img_crop.shape[0], img_crop.shape[1]
    window_len_new = max(H_new, W_new)
    pad_updown_new = int((window_len_new - H_new)/2)
    pad_leftright_new = int((window_len_new - W_new)/2)

    # ReUpdate joint points and center (because of the padding)
    ary_pts_crop = np.where(ary_pts_crop == [0, 0, 0], ary_pts_crop, ary_pts_crop + np.array([pad_leftright_new, pad_updown_new, 0]))
    c_crop = c_crop + np.array([pad_leftright_new, pad_updown_new])

    img_crop = cv2.copyMakeBorder(img_crop, pad_updown_new, pad_updown_new, pad_leftright_new, pad_leftright_new, cv2.BORDER_CONSTANT, value=0)

    # change dtype and num scale
    img_crop = img_crop / 255.
    img_crop = img_crop.astype(np.float)

    if use_flip:
        flip = np.random.random() > 0.5
        if flip:
            # (H,W,C)
            img_crop = np.flip(img_crop, 1)
            # Calculate flip pts, remember to filter [0,0] which is no available heatmap
            ary_pts_crop = np.where(ary_pts_crop == [0, 0, 0], ary_pts_crop,
                                    [window_len_new, 0, 0] + ary_pts_crop * [-1, 1, 0])
            c_crop = [window_len_new, 0] + c_crop * [-1, 1]
            # Rearrange pts
            # because of flip, left->right, right->left
            ary_pts_crop = np.concatenate((ary_pts_crop[5::-1], ary_pts_crop[6:10], ary_pts_crop[15:9:-1]))

    return img_crop, ary_pts_crop, c_crop


def change_resolu(img, pts, c, resolu_out):
    '''
    :param img: np array of the origin image
    :param pts: joint points np array corresponding to the image, same resolu as img
    :param c: center
    :param resolu_out: a list or tuple
    :return: img_out, pts_out, c_out under resolu_out
    '''
    H_in = img.shape[0]
    W_in = img.shape[1]
    H_out = resolu_out[0]
    W_out = resolu_out[1]
    H_scale = H_in/H_out
    W_scale = W_in/W_out

    pts_out = pts/np.array([W_scale, H_scale, 1])
    c_out = c/np.array([W_scale, H_scale])
    img_out = skimage.transform.resize(img, resolu_out)

    return img_out, pts_out, c_out


def show_stack_joints(img, pts, c=[0, 0], draw_lines=True, num_fig=1):
    '''
    Not support batch 
    :param img: np array, (H,W,C)
    :param pts: same resolu as img, joint points, np array (16,3)
    :param c: center, np array (2,)
    '''
    # In case pts is not np array
    pts = np.array(pts)
    dict_style = {
        0: 'origin img',

        1: ['left ankle', 'b', 'x'],
        2: ['left knee', 'b', '^'],
        3: ['left hip', 'b', 'o'],

        4: ['right hip', 'r', 'o'],
        5: ['right knee', 'r', '^'],
        6: ['right ankle', 'r', 'x'],

        7: ['belly', 'y', 'o'],
        8: ['chest', 'y', 'o'],
        9: ['neck', 'y', 'o'],
        10: ['head', 'y', '*'],

        11: ['left wrist', 'b', 'x'],
        12: ['left elbow', 'b', '^'],
        13: ['left shoulder', 'b', 'o'],

        14: ['right shoulder', 'r', 'o'],
        15: ['right elbow', 'r', '^'],
        16: ['right wrist', 'r', 'x']
    }
    plt.figure(num_fig)
    plt.imshow(img)
    list_pt_H, list_pt_W = [], []
    list_pt_cH, list_pt_cW = [], []
    for i in range(pts.shape[0]):
        list_pt_W.append(pts[i, 0])  # x axis
        list_pt_H.append(pts[i, 1])  # y axis
    list_pt_cW.append(c[0])
    list_pt_cH.append(c[1])
    for i in range(pts.shape[0]):
        plt.scatter(list_pt_W[i], list_pt_H[i], color=dict_style[i+1][1], marker=dict_style[i+1][2])
    plt.scatter(list_pt_cW, list_pt_cH, color='b', marker='*')
    if draw_lines:                                
        # Body
        plt.plot(list_pt_W[6:10], list_pt_H[6:10], color='y', linewidth=2)
        plt.plot(list_pt_W[2:4], list_pt_H[2:4], color='y', linewidth=2)
        plt.plot(list_pt_W[12:14], list_pt_H[12:14], color='y', linewidth=2)
        # Left arm
        plt.plot(list_pt_W[10:13], list_pt_H[10:13], color='b', linewidth=2)
        # Right arm
        plt.plot(list_pt_W[13:16], list_pt_H[13:16], color='r', linewidth=2)
        # Left leg
        plt.plot(list_pt_W[0:3], list_pt_H[0:3], color='b', linewidth=2)
        # Right leg
        plt.plot(list_pt_W[3:6], list_pt_H[3:6], color='r', linewidth=2)
    plt.axis('off')
    plt.show()




def show_heatmaps(img, heatmaps, c=np.zeros((2)), num_fig=1):
    '''
    :param img: np array (H,W,3)
    :param heatmaps: np array (H,W,num_pts)
    :param c: center, np array (2,)
    '''
    H, W = img.shape[0], img.shape[1]
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

    # resize heatmap to size of image
    if heatmaps.shape[0] != H:
        heatmaps = skimage.transform.resize(heatmaps, (H, W))

    heatmap_c = generate_heatmap(np.zeros((H, W)), c)
    plt.figure(num_fig)
    for i in range(heatmaps.shape[2] + 1):
        plt.subplot(4, 5, i + 1)
        plt.title(dict_name[i])
        if i == 0:
            plt.imshow(img)
        else:
            plt.imshow(heatmaps[:, :, i - 1])
        plt.axis('off')
    plt.subplot(4, 5, 20)
    plt.imshow(heatmap_c)  # Only take in (H,W) or (H,W,3)
    plt.axis('off')
    plt.show()