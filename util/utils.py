"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import torch.nn as nn
import torch.nn.init as init
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
from tqdm import tqdm
from scipy.ndimage import map_coordinates
from skimage.transform._geometric import GeometricTransform
from skimage.transform import warp, warp_coords, AffineTransform
# from skimage.transform import PiecewiseAffineTransform, warp, warp_coords, AffineTransform
from scipy.optimize import linprog
from scipy.spatial import Delaunay, ConvexHull


def check_directory_and_create(dir_path, exists_warning=False):
    """
    Checks if the path specified is a directory or creates it if it doesn't
    exist.
    
    Args:
        dir_path (string): directory path to check/create
    
    Returns:
        (string): the input path
    """
    if os.path.exists(dir_path):
        if not os.path.isdir(dir_path):
            raise ValueError(f"Given path {dir_path} is not a directory")
        elif exists_warning:
            print(f"WARNING: Already existing experiment folder {dir_path}."
                  "It is recommended to change experiment_id in "
                  "configs/exp_config.json file. Proceeding by overwriting")
    else:
        os.makedirs(dir_path)
    return os.path.abspath(dir_path)
    

def in_hull(points, x):
    """
    Function which checks if a given set of points 'points' belongs in the
    convex hull defined by the vertices 'x'
    Args:
        points (n_points, dim): np.ndarray
        x (dim): np.ndarrady
    """
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class ShapeParts:
    def __init__(self, np_pts):
        self.data = np_pts

    def part(self, idx):
        return Point(self.data[idx, 0], self.data[idx, 1])


class Record():
    def __init__(self, type_list):
        self.data, self.count = {}, {}
        self.type_list = type_list
        self.max_min_data = None
        for t in type_list:
            self.data[t] = 0.0
            self.count[t] = 0.0

    def add(self, new_data, c=1.0):
        for t in self.type_list:
            self.data[t] += new_data
            self.count[t] += c

    def per(self, t):
        return self.data[t] / (self.count[t] + 1e-32)

    def clean(self, t):
        self.data[t], self.count[t] = 0.0, 0.0

    def is_better(self, t, greater):
        if(self.max_min_data == None):
            self.max_min_data = self.data[t]
            return True
        else:
            if(greater):
                if(self.data[t] > self.max_min_data):
                    self.max_min_data = self.data[t]
                    return True
            else:
                if (self.data[t] < self.max_min_data):
                    self.max_min_data = self.data[t]
                    return True
        return False

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def vis_landmark_on_img(img, shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''
    if (type(shape) == ShapeParts):
        def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
            for i in idx_list:
                cv2.line(img, (shape.part(i).x, shape.part(i).y), (shape.part(i + 1).x, shape.part(i + 1).y),
                         color, lineWidth)
            if (loop):
                cv2.line(img, (shape.part(idx_list[0]).x, shape.part(idx_list[0]).y),
                         (shape.part(idx_list[-1] + 1).x, shape.part(idx_list[-1] + 1).y), color, lineWidth)

        draw_curve(list(range(0, 16)))  # jaw
        draw_curve(list(range(17, 21)), color=(0, 0, 255))  # eye brow
        draw_curve(list(range(22, 26)), color=(0, 0, 255))
        draw_curve(list(range(27, 35)))  # nose
        draw_curve(list(range(36, 41)), loop=True)  # eyes
        draw_curve(list(range(42, 47)), loop=True)
        draw_curve(list(range(48, 59)), loop=True, color=(0, 255, 255))  # mouth
        draw_curve(list(range(60, 67)), loop=True, color=(255, 255, 0))

    else:
        def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
            for i in idx_list:
                cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
            if (loop):
                cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                         (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

        draw_curve(list(range(0, 16)))  # jaw
        draw_curve(list(range(17, 21)), color=(0, 0, 255))  # eye brow
        draw_curve(list(range(22, 26)), color=(0, 0, 255))
        draw_curve(list(range(27, 35)))  # nose
        draw_curve(list(range(36, 41)), loop=True)  # eyes
        draw_curve(list(range(42, 47)), loop=True)
        draw_curve(list(range(48, 59)), loop=True, color=(0, 255, 255))  # mouth
        draw_curve(list(range(60, 67)), loop=True, color=(255, 255, 0))

    return img


def vis_landmark_on_plt(fl,  x_offset=0.0, show_now=True, c='r'):
    def draw_curve(shape, idx_list, loop=False, x_offset=0.0, c=None):
        for i in idx_list:
            plt.plot((shape[i, 0] + x_offset, shape[i + 1, 0] + x_offset), (-shape[i, 1], -shape[i + 1, 1]), c=c, lineWidth=1)
        if (loop):
            plt.plot((shape[idx_list[0], 0] + x_offset, shape[idx_list[-1] + 1, 0] + x_offset),
                     (-shape[idx_list[0], 1], -shape[idx_list[-1] + 1, 1]), c=c, lineWidth=1)

    draw_curve(fl, list(range(0, 16)), x_offset=x_offset, c=c)  # jaw
    draw_curve(fl, list(range(17, 21)), x_offset=x_offset, c=c)  # eye brow
    draw_curve(fl, list(range(22, 26)), x_offset=x_offset, c=c)
    draw_curve(fl, list(range(27, 35)), x_offset=x_offset, c=c)  # nose
    draw_curve(fl, list(range(36, 41)), loop=True, x_offset=x_offset, c=c)  # eyes
    draw_curve(fl, list(range(42, 47)), loop=True, x_offset=x_offset, c=c)
    draw_curve(fl, list(range(48, 59)), loop=True, x_offset=x_offset, c=c)  # mouth
    draw_curve(fl, list(range(60, 67)), loop=True, x_offset=x_offset, c=c)

    if(show_now):
        plt.show()


def try_mkdir(dir):
    try:
        os.mkdir(dir)
    except:
        pass

import numpy
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


def get_puppet_info(DEMO_CH, ROOT_DIR):
    import numpy as np
    B = 5000
    # for wilk example
    if (DEMO_CH == 'wilk_old'):
        bound = np.array([-B, -B, -B, 459, -B, B+918, 419, B+918, B+838, B+918, B+838, 459, B+838, -B, 419, -B]).reshape(1, -1)
        # bound = np.array([0, 0, 0, 459, 0, 918, 419, 918, 838, 918, 838, 459, 838, 0, 419, 0]).reshape(1, -1)
        scale, shift = -0.005276414887140783, np.array([-475.4316, -193.53225])
    elif (DEMO_CH == 'sketch'):
        bound = np.array([-10000, -10000, -10000, 221, -10000, 10443, 232, 10443, 10465, 10443, 10465, 221, 10465, -10000, 232, -10000]).reshape(1, -1)
        scale, shift = -0.006393177201290783, np.array([-226.8411, -176.5216])
    elif (DEMO_CH == 'onepunch'):
        bound = np.array([0, 0, 0, 168, 0, 337, 282, 337, 565, 337, 565, 168, 565, 0, 282, 0]).reshape(1, -1)
        scale, shift = -0.007558707536598317, np.array([-301.4903, -120.05265])
    elif (DEMO_CH == 'cat'):
        bound = np.array([0, 0, 0, 315, 0, 631, 299, 631, 599, 631, 599, 315, 599, 0, 299, 0]).reshape(1, -1)
        scale, shift = -0.009099476040795225, np.array([-297.17085, -259.2363])
    elif (DEMO_CH == 'paint'):
        bound = np.array([0, 0, 0, 249, 0, 499, 212, 499, 424, 499, 424, 249, 424, 0, 212, 0]).reshape(1, -1)
        scale, shift = -0.007409177996872789, np.array([-161.92345878, -249.40250103])
    elif (DEMO_CH == 'mulaney'):
        bound = np.array([0, 0, 0, 255, 0, 511, 341, 511, 682, 511, 682, 255, 682, 0, 341, 0]).reshape(1, -1)
        scale, shift = -0.010651548568731444, np.array([-333.54245, -189.081])
    elif (DEMO_CH == 'cartoonM_old'):
        bound = np.array([0, 0, 0, 299, 0, 599, 399, 599, 799, 599, 799, 299, 799, 0, 399, 0]).reshape(1, -1)
        scale, shift = -0.0055312373170456845, np.array([-398.6125, -240.45235])
    elif (DEMO_CH == 'beer'):
        bound = np.array([0, 0, 0, 309, 0, 618, 260, 618, 520, 618, 520, 309, 520, 0, 260, 0]).reshape(1, -1)
        scale, shift = -0.0054102709937112374, np.array([-254.1478, -156.6971])
    elif (DEMO_CH == 'color'):
        bound = np.array([0, 0, 0, 140, 0, 280, 249, 280, 499, 280, 499, 140, 499, 0, 249, 0]).reshape(1, -1)
        scale, shift = -0.012986159189209149, np.array([-237.27065, -79.2465])
    else:
        if (os.path.exists(os.path.join(ROOT_DIR, DEMO_CH + '.jpg'))):
            img = cv2.imread(os.path.join(ROOT_DIR, DEMO_CH + ".jpg"))
        elif (os.path.exists(os.path.join(ROOT_DIR, DEMO_CH + '.png'))):
            img = cv2.imread(os.path.join(ROOT_DIR, DEMO_CH + ".png"))
        else:
            print('not file founded.')
            exit(0)
        size = img.shape
        h = size[1] - 1
        w = size[0] - 1
        bound = np.array([-B, -B,
                          -B, w//4,
                          -B, w // 2,
                          -B, w//4*3,
                          -B, B + w,
                          h // 2, B+w,
                          B+h, B+w,
                          B+h, w // 2,
                          B+h, -B,
                          h//4, -B,
                          h // 2, -B,
                          h//4*3, -B]).reshape(1, -1)
        ss = np.loadtxt(os.path.join(ROOT_DIR, DEMO_CH + '_scale_shift.txt'))
        scale, shift = ss[0], np.array([ss[1], ss[2]])

    return bound, scale, shift


def close_input_face_mouth(shape_3d, p1=0.7, p2=0.5):
    shape_3d = shape_3d.reshape((1, 68, 3))
    index1 = list(range(60 - 1, 55 - 1, -1))
    index2 = list(range(68 - 1, 65 - 1, -1))
    mean_out = 0.5 * (shape_3d[:, 49:54] + shape_3d[:, index1])
    mean_in = 0.5 * (shape_3d[:, 61:64] + shape_3d[:, index2])
    shape_3d[:, 50:53] -= (shape_3d[:, 61:64] - mean_in) * p1
    shape_3d[:, list(range(59 - 1, 56 - 1, -1))] -= (shape_3d[:, index2] - mean_in) * p1
    shape_3d[:, 49] -= (shape_3d[:, 61] - mean_in[:, 0]) * p2
    shape_3d[:, 53] -= (shape_3d[:, 63] - mean_in[:, -1]) * p2
    shape_3d[:, 59] -= (shape_3d[:, 67] - mean_in[:, 0]) * p2
    shape_3d[:, 55] -= (shape_3d[:, 65] - mean_in[:, -1]) * p2
    # shape_3d[:, 61:64] = shape_3d[:, index2] = mean_in
    shape_3d[:, 61:64] -= (shape_3d[:, 61:64] - mean_in) * p1
    shape_3d[:, index2] -= (shape_3d[:, index2] - mean_in) * p1
    shape_3d = shape_3d.reshape((68, 3))

    return shape_3d

def norm_input_face(shape_3d):
    scale = 1.6 / (shape_3d[0, 0] - shape_3d[16, 0])
    shift = - 0.5 * (shape_3d[0, 0:2] + shape_3d[16, 0:2])
    shape_3d[:, 0:2] = (shape_3d[:, 0:2] + shift) * scale
    face_std = np.loadtxt('src/dataset/utils/STD_FACE_LANDMARKS.txt').reshape(68, 3)
    shape_3d[:, -1] = face_std[:, -1] * 0.1
    shape_3d[:, 0:2] = -shape_3d[:, 0:2]

    return shape_3d, scale, shift

def add_naive_eye(fl):
    for t in range(fl.shape[0]):
        r = 0.95
        fl[t, 37], fl[t, 41] = r * fl[t, 37] + (1 - r) * fl[t, 41], (1 - r) * fl[t, 37] + r * fl[t, 41]
        fl[t, 38], fl[t, 40] = r * fl[t, 38] + (1 - r) * fl[t, 40], (1 - r) * fl[t, 38] + r * fl[t, 40]
        fl[t, 43], fl[t, 47] = r * fl[t, 43] + (1 - r) * fl[t, 47], (1 - r) * fl[t, 43] + r * fl[t, 47]
        fl[t, 44], fl[t, 46] = r * fl[t, 44] + (1 - r) * fl[t, 46], (1 - r) * fl[t, 44] + r * fl[t, 46]

    K1, K2 = 10, 15
    length = fl.shape[0]
    close_time_stamp = [30]
    t = 30
    while (t < length - 1 - K2):
        t += 60
        t += np.random.randint(30, 90)
        if (t < length - 1 - K2):
            close_time_stamp.append(t)
    for t in close_time_stamp:
        fl[t, 37], fl[t, 41] = 0.25 * fl[t, 37] + 0.75 * fl[t, 41], 0.25 * fl[t, 37] + 0.75 * fl[t, 41]
        fl[t, 38], fl[t, 40] = 0.25 * fl[t, 38] + 0.75 * fl[t, 40], 0.25 * fl[t, 38] + 0.75 * fl[t, 40]
        fl[t, 43], fl[t, 47] = 0.25 * fl[t, 43] + 0.75 * fl[t, 47], 0.25 * fl[t, 43] + 0.75 * fl[t, 47]
        fl[t, 44], fl[t, 46] = 0.25 * fl[t, 44] + 0.75 * fl[t, 46], 0.25 * fl[t, 44] + 0.75 * fl[t, 46]

        def interp_fl(t0, t1, t2, r):
            for index in [37, 38, 40, 41, 43, 44, 46, 47]:
                fl[t0, index] = r * fl[t1, index] + (1 - r) * fl[t2, index]

        for t0 in range(t - K1 + 1, t):
            interp_fl(t0, t - K1, t, r=(t - t0) / 1. / K1)
        for t0 in range(t + 1, t + K2):
            interp_fl(t0, t, t + K2, r=(t + K2 - 1 - t0) / 1. / K2)

    return fl


def read_line(line, dtype=float):
    """
    Function which reads a line separated by ' ' e.g '12 24 46' and returns
    a list as [12, 24, 46]
    """
    return list(map(dtype, line.split(' ')))


def list2array(list):
    return np.asarray(list)


def read_txt(txt_path, dtype):
    txt_lines = []
    with open(txt_path, "r") as fd:
        # lines = fd.read()
        for line in fd:
            print(line)
            txt_lines.append(list2array(read_line(line, dtype)))
    return txt_lines


def _merge_images(img_top, img_bottom, mask=0):
    """
    Function to combine two images with mask by replacing all pixels
    of img_bottom which equals to mask by pixels from img_top.

    script from
    https://github.com/marsbroshok/face-replace/blob/master/faceWarp.py

    :param img_top: greyscale image which will replace masked pixels
    :param img_bottom: greyscale image which pixels will be replaced
    :param mask: pixel value to be used as mask (int)
    :return: combined greyscale image
    """
    img_top = skimage.img_as_ubyte(img_top)
    img_bottom = skimage.img_as_ubyte(img_bottom)
    merge_layer = img_top == mask
    img_top[merge_layer] = img_bottom[merge_layer]
    return img_top


def face_warp(src_face,
              src_face_lm,
              dst_face_lm,
              bg,
              warp_only=False,
              use_bg=True):
    """
    Function takes two faces and landmarks and warp one face around another
    according to the face landmarks.

    script modified from
    https://github.com/marsbroshok/face-replace/blob/master/faceWarp.py


    :param src_face: grayscale (?) image (np.array of int) of face
        which will warped around second face
    :param src_face_lm: landmarks for the src_face
    :param dst_face: predicted image landmarks (np.array of int) which will
        be replaced by src_face.
    :param bg: image background
    :return: image with warped face
    """
    # Helpers
    output_shape = src_face.shape[:2]  # dimensions of our final image (from webcam eg)

    # Get the landmarks/parts for the face.
    # try:
    # dst_face_lm = find_landmarks(dst_face, predictor, opencv_facedetector=False)
    # src_face_coord = _shape_to_array(src_face_lm)
    # dst_face_coord = _shape_to_array(dst_face_lm) + 10 * np.random.rand(68, 2)

    # dst_face_lm = find_landmarks(dst_face, predictor, opencv_facedetector=False)
    src_face_coord = src_face_lm
    dst_face_coord = dst_face_lm

    warp_trans = PiecewiseAffineTransform()
    # warp_trans.estimate(dst_face_coord, src_face_coord)
    # might be buggy here and need src_face, dst_face instead !!!!!
    # warp_trans.estimate(dst_face_coord, src_face_coord)
    warp_trans.estimate(src_face_coord, dst_face_coord)
    warped_face = skimage.transform.warp(src_face,
                                         warp_trans) #, output_shape=output_shape)
    # except:
    #     warped_face = dst_face

    # Merge two images: new warped face and background of dst image
    # through using a mask (pixel level value is 0)

    # this might need to be investigated too!!!!
    if not warp_only:
        if use_bg:
            warped_face = _merge_images(warped_face, bg)
        else:
            warped_face = _merge_images(warped_face, src_face)
    return warped_face


def face_warp_coord(src_face,
                    src_face_lm,
                    dst_face_lm,
                    tri,
                    bg,
                    warp_only=False,
                    use_bg=True):
    """
    Function takes two faces and landmarks and warp one face around another
    according to the face landmarks.

    script modified from
    https://github.com/marsbroshok/face-replace/blob/master/faceWarp.py


    :param src_face: grayscale (?) image (np.array of int) of face
        which will warped around second face
    :param src_face_lm: landmarks for the src_face
    :param dst_face: predicted image landmarks (np.array of int) which will
        be replaced by src_face.
    :param bg: image background
    :return: image with warped face
    """
    src_face_coord = src_face_lm
    dst_face_coord = dst_face_lm

    affines = []
    # find affine mapping from source positions to destination
    for k in tri:
        affine = AffineTransform()
        affine.estimate(src_face_coord[k, :], dst_face_coord[k, :])
        affines.append(affine)

    inverse_affines = []
    # find the inverse affine mapping
    for k in tri:
        affine = AffineTransform()
        affine.estimate(dst_face_coord[k, :], src_face_coord[k, :])
        inverse_affines.append(affine)
    


    coords = warp_coords(coord_map, src_face.shape)
    warped_face = map_coordinates(src_face, coords)
    if not warp_only:
        if use_bg:
            warped_face = _merge_images(warped_face, bg)
        else:
            warped_face = _merge_images(warped_face, src_face)
    return warped_face


class PiecewiseAffineTransformTriang(GeometricTransform):
    """2D piecewise affine transformation based on given triangulation.
    Control points are used to define the mapping. The transform is based on
    a Delaunay triangulation of the points to form a mesh. Each triangle is
    used to find a local affine transform.
    Attributes
    ----------
    affines : list of AffineTransform objects
        Affine transformations for each triangle in the mesh.
    inverse_affines : list of AffineTransform objects
        Inverse affine transformations for each triangle in the mesh.
    """

    def __init__(self, hulls, simplex, reshape=False):
        self._tesselation = None
        self._inverse_tesselation = None
        self.affines = None
        self.inverse_affines = None
        # originally simplex has size MxN but reshape to M*N
        self.simplex = simplex
        if reshape:
            self.simplex = simplex.reshape(-1,)
        self.hulls = hulls

    def estimate(self, src, dst, delaunay):
        """Estimate the transformation from a set of corresponding points.
        Number of source and destination coordinates must match.
        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.
        delaunay : (*, 3) array.
            The given triangulation
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        # forward piecewise affine for a given triangulation 
        self._tesselation = delaunay

        # import pdb; pdb.set_trace()
        # find affine mapping from source positions to destination
        self.affines = []
        for tri in self._tesselation:
            print(tri)
            print(src[tri], dst[tri])
            affine = AffineTransform()
            affine.estimate(src[tri], dst[tri])
            self.affines.append(affine)
        # import pdb; pdb.set_trace()
        # inverse piecewise affine
        # keep the same trianglulation
        # find affine mapping from source positions to destination
        self._inverse_tesselation = delaunay
        self.inverse_affines = []
        for tri in self._inverse_tesselation:
            affine = AffineTransform()
            affine.estimate(dst[tri], src[tri])
            self.inverse_affines.append(affine)

        return True

    def __call__(self, coords):
        """Apply forward transformation.
        Coordinates outside of the mesh will be set to `- 1`.
        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.
        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.
        """

        out = np.empty_like(coords, np.double)

        # determine triangle index for each coordinate
        # import pdb; pdb.set_trace()
        coords = coords.astype(int)
        simplex = self.simplex

        # coordinates outside of mesh
        out[simplex == -1, :] = -1

        for index in range(len(self.hulls)):
            # affine transform for triangle - convex hull
            affine = self.affines[index]
            # all coordinates within triangle - convex hull
            index_mask = simplex == index

            out[index_mask, :] = affine(coords[index_mask, :])

        return out

    def inverse(self, coords):
        """Apply inverse transformation.
        Coordinates outside of the mesh will be set to `- 1`.
        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.
        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.
        """

        out = np.empty_like(coords, np.double)

        # determine triangle index for each coordinate
        simplex = self.simplex

        # coordinates outside of mesh
        out[simplex == -1, :] = -1

        for index in range(len(self.hulls)):
            # affine transform for triangle
            affine = self.inverse_affines[index]
            # all coordinates within triangle
            index_mask = simplex == index

            out[index_mask, :] = affine(coords[index_mask, :])

        return out


def point_in_hull(point, hull, tolerance=1e-12):
    """
    Function which calculates the dot product between a given point and
    the normal equations of convex hull. Tollerance is used for numerical
    stability purposes.
    Args:
        point (dim,): a point with dim dimensiolaty
        hull (scipy.ConvexHull object): convex hull
        tollerance (float): tollerance in calculations
    Output:
        bool: belongs or not to the convex hull
    """
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def get_simplex(src_img, src_land, triangulation):
    """
    Function which maps the points of an image to a simplex
    based on a given triangulation
    """
    # list of convex hulls, i.e triangles in the source image
    hulls = []
    for tri in triangulation:
        hull = ConvexHull(src_land[tri, :])
        hulls.append(hull)

    print(f"constructed {len(hulls)} triangles")
    # iterate over image to get the simplex of every point
    rows, cols = src_img.shape[0], src_img.shape[1]
    import pdb; pdb.set_trace()
    simplex_matrix = np.empty((rows, cols), int)
    for i in tqdm(range(rows)):
        for j in range(cols):
            pt = np.array([i, j])
            for h_id, h in enumerate(hulls):
                if point_in_hull(pt, h):
                    # if h_id in [0,1,2]:
                    #     print(f"point {i},{j} belongs to hull {h_id}")
                    simplex_matrix[i][j] = h_id
                    break

    return hulls, simplex_matrix

class PiecewiseAffineTransform(GeometricTransform):
    """2D piecewise affine transformation.

    Control points are used to define the mapping. The transform is based on
    a Delaunay triangulation of the points to form a mesh. Each triangle is
    used to find a local affine transform.

    Attributes
    ----------
    affines : list of AffineTransform objects
        Affine transformations for each triangle in the mesh.
    inverse_affines : list of AffineTransform objects
        Inverse affine transformations for each triangle in the mesh.

    """

    def __init__(self):
        self._tesselation = None
        self._inverse_tesselation = None
        self.affines = None
        self.inverse_affines = None

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        # forward piecewise affine
        # triangulate input positions into mesh
        self._tesselation = Delaunay(src)
        # find affine mapping from source positions to destination
        self.affines = []
        for tri in self._tesselation.vertices:
            affine = AffineTransform()
            affine.estimate(src[tri, :], dst[tri, :])
            self.affines.append(affine)

        # inverse piecewise affine
        # triangulate input positions into mesh
        self._inverse_tesselation = Delaunay(dst)
        # find affine mapping from source positions to destination
        self.inverse_affines = []
        for tri in self._inverse_tesselation.vertices:
            affine = AffineTransform()
            affine.estimate(dst[tri, :], src[tri, :])
            self.inverse_affines.append(affine)

        return True

    def __call__(self, coords):
        """Apply forward transformation.

        Coordinates outside of the mesh will be set to `- 1`.

        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        """

        out = np.empty_like(coords, np.double)

        # determine triangle index for each coordinate
        simplex = self._tesselation.find_simplex(coords)
        # import pdb; pdb.set_trace()

        # coordinates outside of mesh
        out[simplex == -1, :] = -1

        for index in range(len(self._tesselation.vertices)):
            # affine transform for triangle
            affine = self.affines[index]
            # all coordinates within triangle
            index_mask = simplex == index

            out[index_mask, :] = affine(coords[index_mask, :])

        return out

    def inverse(self, coords):
        """Apply inverse transformation.

        Coordinates outside of the mesh will be set to `- 1`.

        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        """

        out = np.empty_like(coords, np.double)

        # determine triangle index for each coordinate
        simplex = self._inverse_tesselation.find_simplex(coords)

        # coordinates outside of mesh
        out[simplex == -1, :] = -1

        for index in range(len(self._inverse_tesselation.vertices)):
            # affine transform for triangle
            affine = self.inverse_affines[index]
            # all coordinates within triangle
            index_mask = simplex == index

            out[index_mask, :] = affine(coords[index_mask, :])

        return out