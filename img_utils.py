import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage.transform
import skimage.filters


def load_img(dir):
    img = mpimg.imread(dir)
    return img


def generate_heatmap(heatmap, pt, sigma_valu=2):
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



def generate_heatmaps(img, pts, sigma_valu=2):
    '''
    Generate 16 heatmaps
    :param img: np arrray img, (H,W,C)
    :param pts: joint points coords, np array, same resolu as img
    :param sigma: should be a tuple with odd values (obsolete)
    :param sigma_valu: vaalue for gaussian blur
    :return: np array heatmaps, (H,W,num_pts)
    '''
    H, W = img.shape[0], img.shape[1]
    num_pts = len(pts)
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

    heatmaps[:, :, num_pts] = 1.0 - np.max(heatmaps[:, :, :num_pts], axis=2) # add background dim

    return heatmaps



def crop(img, ele_anno, use_rotate=True, use_hflip=False, crop_size=368):

    # get bbox
    pts = ele_anno['joint_self']
    cen = ele_anno['objpos'].copy()

    pts = np.array(pts)
    pts_nonzero = np.where(pts[:,1] != 0)[0]
    pts_zero = np.where(pts[:,1] == 0)[0]
    xs = pts[:, 0]
    #xs = pts[pts_nonzero][:, 0]
    ys = pts[:, 1]
    #ys = pts[pts_nonzero][:, 1]
    bbox = [(max(max(xs[pts_nonzero]),cen[0]) + min(min(xs[pts_nonzero]), cen[0]) )/2.0,
            (max(max(ys[pts_nonzero]),cen[1]) + min(min(ys[pts_nonzero]), cen[1]) )/2.0,
            (max(max(xs[pts_nonzero]),cen[0]) - min(min(xs[pts_nonzero]), cen[0]) )*1.3,
            (max(max(ys[pts_nonzero]),cen[1]) - min(min(ys[pts_nonzero]), cen[1]) )*1.3]
    bbox = np.array(bbox)

    if use_rotate:
        H, W = img.shape[0], img.shape[1]
        img_center = (W / 2.0 , H / 2.0)
        degree = np.random.uniform(-30,30)
        rotateMat = cv2.getRotationMatrix2D(img_center, degree, scale=1.0)
        cos_val = np.abs(rotateMat[0, 0])
        sin_val = np.abs(rotateMat[0, 1])
        new_width = int(H * sin_val + W * cos_val)
        new_height = int(H * cos_val + W * sin_val)
        rotateMat[0, 2] += (new_width / 2.) - img_center[0]
        rotateMat[1, 2] += (new_height / 2.) - img_center[1]

        img = cv2.warpAffine(img, rotateMat, (new_width, new_height), borderValue=(0,0,0)) # black border
        

        num = len(pts)
        for i in range(num):
            if pts[i,1]==0:
                continue
            x = xs[i]
            y = ys[i]
            p = np.array([x, y, 1])
            p = rotateMat.dot(p)
            xs[i] = p[0]
            ys[i] = p[1]

        x = cen[0]
        y = cen[1]
        p = np.array([x, y, 1])
        p = rotateMat.dot(p)
        cen[0] = p[0]
        cen[1] = p[1]
        # update bbox 
        bbox = [(max(max(xs[pts_nonzero]),cen[0]) + min(min(xs[pts_nonzero]), cen[0]) )/2.0,
                (max(max(ys[pts_nonzero]),cen[1]) + min(min(ys[pts_nonzero]), cen[1]) )/2.0,
                (max(max(xs[pts_nonzero]),cen[0]) - min(min(xs[pts_nonzero]), cen[0]) )*1.3,
                (max(max(ys[pts_nonzero]),cen[1]) - min(min(ys[pts_nonzero]), cen[1]) )*1.3]
        bbox = np.array(bbox)


    ###
    #pts = [[xs[i], ys[i], pts[i,2]] for i in range(len(xs))]
    #show_stack_joints(img, pts, cen)
    ###

    H,W = img.shape[0], img.shape[1]
    scale = np.random.uniform(0.8, 1.3) # given data
    bbox[2] *= scale
    bbox[3] *= scale
    # topleft:x1,y1  bottomright:x2,y2
    bb_x1 = int(bbox[0] - bbox[2]/2)
    bb_y1 = int(bbox[1] - bbox[3]/2)
    bb_x2 = int(bbox[0] + bbox[2]/2)
    bb_y2 = int(bbox[1] + bbox[3]/2)

    if bb_x1<0 or bb_x2>W or bb_y1<0 or bb_y2>H:
        pad = int(max(-bb_x1, bb_x2-W, -bb_y1, bb_y2-H))
        img = np.pad(img, ((pad, pad),(pad,pad),(0,0)), 'constant')
    else:
        pad = 0
    img = img[bb_y1+pad:bb_y2+pad, bb_x1+pad:bb_x2+pad]

    xs = np.where(xs != 0, xs-bb_x1, xs)
    ys = np.where(ys != 0, ys-bb_y1, ys)
    #ys = np.array([ys[i]-bb_y1 for i in range(len(ys)) if i in pts_nonzero])
    bbox[0] -= bb_x1
    bbox[1] -= bb_y1
    
    cen[0] -= bb_x1
    cen[1] -= bb_y1

    # horizontal flip
    if use_hflip and np.random.rand() > 0.:
        H,W = img.shape[0], img.shape[1]
        img = cv2.flip(img, 1)
        xs = (W - 1) - xs
        cen[0] = (W - 1) - cen[0]
        for i,j in ((12,13),(11,14),(10,15),(2,3),(1,4), (0,5)):
            xs[i], xs[j] = xs[j].copy(), xs[i].copy()
            ys[i], ys[j] = ys[j].copy(), ys[i].copy()

    # resize
    H,W = img.shape[0], img.shape[1]
    xs = xs*crop_size/W
    ys = ys*crop_size/H
    cen[0] = cen[0]*crop_size/W
    cen[1] = cen[1]*crop_size/H
    img = cv2.resize(img, (crop_size, crop_size))

    # generate heatmaps


    # generate centermap

    # pts_crop = []
    # for i in range(pts.shape[0]):
    #     if i in pts_nonzero:
    #         pts_crop.append([xs[i], ys[i], 1.0])
    #     else:
    #         pts_crop.append([ 0.,0.,0. ])
    # pts_crop = np.array(pts_crop)

    pts = [[xs[i], ys[i], pts[i,2]] for i in range(len(xs))]

    return img, pts, cen



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


# TODO: Modify flaws in showing joints
# Chest and nest often overlap
def show_stack_joints(img, pts, c=[0, 0], draw_lines=True, num_fig=1):
    '''
    Not support batch 
    :param img: np array, (H,W,C)
    :param pts: same resolu as img, joint points, np array (16,3) or (16,2)
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
        # change: if left ankle or knee doesn't exist
        if pts[0,1] != 0:
            if pts[1,1] != 0:
                plt.plot(list_pt_W[0:3], list_pt_H[0:3], color='b', linewidth=2)
            else:
                pass  # werid condition
        else:
            if pts[1,1] != 0:
                plt.plot(list_pt_W[1:3], list_pt_H[1:3], color='b', linewidth=2)
            else:
                pass  # draw nothing

        # Right leg
        if pts[5,1] != 0:
            if pts[4,1] != 0:
                plt.plot(list_pt_W[3:6], list_pt_H[3:6], color='r', linewidth=2)
            else:
                pass # werid condition
        else:
            if pts[4,1] != 0:
                plt.plot(list_pt_W[3:5], list_pt_H[3:5], color='r', linewidth=2)
            else:
                pass  # draw nothing
    plt.axis('off')
    plt.show()

    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.margins(0,0)
    # plt.savefig('./imgs/preprocess_%d.jpg' %num_fig,bbox_inches='tight',pad_inches=0.0) # remove padding




def show_heatmaps(img, heatmaps, c=np.zeros((2)), num_fig=1):
    '''
    :param img: np array (H,W,3)
    :param heatmaps: np array (H,W,num_pts)
    :param c: center, np array (2,)

    how to deal with negative in heatmaps ??? 
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



def heatmaps_to_coords(heatmaps, resolu_out=(368,368), prob_threshold=0.2):
    '''
    :param heatmaps: (46,46,16)
    :param resolu_out: output resolution list
    :return coord_joints: np array, shape (16,2)
    '''

    num_joints = heatmaps.shape[2]
    # Resize
    heatmaps = cv2.resize(heatmaps, resolu_out)
    print('heatmaps.SHAPE', heatmaps.shape)

    coord_joints = np.zeros((num_joints, 2))
    for i in range(num_joints):
        heatmap = heatmaps[..., i]
        max = np.max(heatmap)
        # Only keep points larger than a threshold
        if max > prob_threshold:
            idx = np.where(heatmap == max)
            H = idx[0][0]
            W = idx[1][0]
        else:
            H = 0
            W = 0
        coord_joints[i] = [W, H]
    return coord_joints