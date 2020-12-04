"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import os, glob
import cv2
import numpy as np
import argparse
import pickle
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
import matplotlib.pyplot as plt
from PIL import Image as im
from util.utils import read_txt, face_warp, face_warp_coord, get_simplex, PiecewiseAffineTransformTriang, _merge_images
from skimage.transform import warp


ADD_NAIVE_EYE = False
GEN_AUDIO = True
GEN_FLS = True

DEMO_CH = 'danbooru1'

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default=DEMO_CH)

parser.add_argument('--load_AUTOVC_name', type=str,
                    default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str,
                    default='examples/ckpt/ckpt_speaker_branch.pth')  #ckpt_audio2landmark_g.pth') #
parser.add_argument('--load_a2l_C_name', type=str,
                    default='examples/ckpt/ckpt_content_branch.pth') #ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str,
                    default='examples/ckpt/ckpt_116_i2i_comb.pth') #ckpt_i2i_finetune_150.pth') #ckpt_image2image.pth') #

parser.add_argument('--amp_lip_x', type=float, default=2.0)
parser.add_argument('--amp_lip_y', type=float, default=2.0)
parser.add_argument('--amp_pos', type=float, default=0.8)
parser.add_argument('--reuse_train_emb_list', default=['45hn7-LXDX8']) 
# other options
#  ['E_kmpT-EfOg']) #  ['E_kmpT-EfOg']) # ['45hn7-LXDX8'])


parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples_cartoon')
parser.add_argument('--approach', type=str, default='custom',
                    choices=["custom", "skimage"],
                    help="custom or skimage available")

#### NEW POSE MODEL
parser.add_argument('--test_end2end', default=True, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--use_wine', default=False, action="store_true")
parser.add_argument('--segment_batch_size', type=int, default=512, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')

opt_parser = parser.parse_args()
# import pdb; pdb.set_trace()

DEMO_CH = opt_parser.jpg

# loads the landmarks for the closed mouth cartoon
shape_3d = np.loadtxt('examples_cartoon/{}_face_close_mouth.txt'.format(opt_parser.jpg))
print(f"the shape of closed mouth landmark image is {shape_3d.shape}")
# data = im.fromarray(shape_3d, 'RGB')
# data.save('temp.png')

''' STEP 3: Generate audio data as input to audio branch '''
au_data = []
ains = glob.glob1('examples', '*.wav')
print(f"the audio files to be converted are {ains}")
ains.sort()
# import pdb; pdb.set_trace()
for ain in ains:
    os.system('ffmpeg -y -loglevel error -i examples/{} -ar 16000 examples/tmp.wav'.format(ain))
    shutil.copyfile('examples/tmp.wav', 'examples/{}'.format(ain))
    print('Processing audio file', ain)
    c = AutoVC_mel_Convertor('examples')
    au_data_i = \
        c.convert_single_wav_to_autovc_input(
            audio_filename=os.path.join('examples', ain),
            autovc_model_path=opt_parser.load_AUTOVC_name)
    au_data += au_data_i # append au_data to list
    # os.remove(os.path.join('examples', 'tmp.wav'))
if(os.path.isfile('examples/tmp.wav')):
    os.remove('examples/tmp.wav')

# camera calibration parameters
fl_data = []
rot_tran, rot_quat, anchor_t_shape = [], [], []
for au, info in au_data:
    au_length = au.shape[0]
    fl = np.zeros(shape=(au_length, 68 * 3))
    fl_data.append((fl, info))
    rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
    rot_quat.append(np.zeros(shape=(au_length, 4)))
    anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))


# remove previous pickles in examples/dump/
if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
if(os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
if (os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

# store the new pickles in examples/dump
with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
    pickle.dump(fl_data, fp)
with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
    pickle.dump(au_data, fp)
with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
    gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
    pickle.dump(gaze, fp)


''' STEP 4: RUN audio->landmark network'''
from src.approaches.train_audio2landmark import Audio2landmark_model
print('Running Audio to Landmark model')
model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
model.test()
print('finish gen fls')

# up to this point there exists a landmark visualization along with the 
# audio samples in examples/


''' STEP 5: de-normalize the output to the original image scale '''
print("Entering Stage 5 of de-normalization of the output to the original image scale")
fls_names = glob.glob1(opt_parser.output_folder, 'pred_fls_*.txt')
fls_names.sort()

for i in range(0,len(fls_names)):
    ains = glob.glob1('examples', '*.wav')
    ains.sort()
    ain = ains[i]
    fl = np.loadtxt(os.path.join(opt_parser.output_folder, fls_names[i])).reshape((-1, 68,3))
    output_dir = os.path.join(opt_parser.output_folder, fls_names[i][:-4])
    try:
        os.makedirs(output_dir)
        print("succesfully created the directory")
    except:
        pass

    print(f"The fl is {fl.shape}")
    # import pdb; pdb.set_trace()
    from util.utils import get_puppet_info

    bound, scale, shift = get_puppet_info(DEMO_CH, ROOT_DIR='examples_cartoon')

    fls = fl.reshape((-1, 68, 3))

    fls[:, :, 0:2] = -fls[:, :, 0:2]
    fls[:, :, 0:2] = (fls[:, :, 0:2] / scale)
    fls[:, :, 0:2] -= shift.reshape(1, 2)

    # fl: N_frames x 68 X 3 ---> fls: N_frames x 204 
    fls = fls.reshape(-1, 68*3)
    # fls = fls.reshape(-1, 204)

    # additional smooth
    from scipy.signal import savgol_filter
    fls[:, 0:48*3] = savgol_filter(fls[:, 0:48*3], 17, 3, axis=0)
    fls[:, 48*3:] = savgol_filter(fls[:, 48*3:], 11, 3, axis=0)
    fls = fls.reshape((-1, 68, 3))

    if (DEMO_CH in ['paint', 'mulaney', 'cartoonM', 'beer', 'color', 'JohnMulaney', 'vangogh', 'jm', 'roy', 'lineface']):
        r = list(range(0, 68))
        fls = fls[:, r, :]
        fls = fls[:, :, 0:2].reshape(-1, 68 * 2)
        fls = np.concatenate((fls, np.tile(bound, (fls.shape[0], 1))), axis=1)
        fls = fls.reshape(-1, 160)

    else:
        r = list(range(0, 48)) + list(range(60, 68))
        fls = fls[:, r, :]
        fls = fls[:, :, 0:2].reshape(-1, 56 * 2)
        fls = np.concatenate((fls, np.tile(bound, (fls.shape[0], 1))), axis=1)
        fls = fls.reshape(-1, 112 + bound.shape[1])

    # warped points contains the 3D landmark for the whole animation
    np.savetxt(os.path.join(output_dir, 'warped_points.txt'), fls, fmt='%.2f')

    # static_points.txt
    # here the open mouth cartoon is taken as the static frame from which
    # the displacements have been calculated
    static_frame = np.loadtxt(os.path.join('examples_cartoon', '{}_face_open_mouth.txt'.format(DEMO_CH)))
    static_frame = static_frame[r, 0:2]
    static_frame = np.concatenate((static_frame, bound.reshape(-1, 2)), axis=0)
    np.savetxt(os.path.join(output_dir, 'reference_points.txt'), static_frame, fmt='%.2f')

    # triangle_vtx_index.txt
    # the pre-calculated delaynay triangulation stored
    #  w.r.t the 68 reference points, landmarks
    # saves the already calculated triangles
    shutil.copy(os.path.join('examples_cartoon', DEMO_CH + '_delauney_tri.txt'),
                os.path.join(output_dir, 'triangulation.txt'))

    os.remove(os.path.join(opt_parser.output_folder,  fls_names[i]))
    ###########################################################################
    ### Translate the network output
    ###########################################################################
    ref_txt = os.path.join(output_dir, "reference_points.txt")
    delaunay_txt = os.path.join(output_dir, "triangulation.txt")
    warped_txt = os.path.join(output_dir, "warped_points.txt")
    cartoon = cv2.imread(os.path.join('examples_cartoon', DEMO_CH+'.png'))
    src_bg = cv2.imread(os.path.join('examples_cartoon', DEMO_CH+'_bg.png'))

    warped_pts = read_txt(warped_txt, dtype=float)
    delaunay_pts = read_txt(delaunay_txt, dtype=int)
    ref_pts = read_txt(ref_txt, dtype=float)

    n_landmarks = len(ref_pts)
    source_cartoon_lm = np.concatenate(ref_pts, axis=0).reshape(n_landmarks, 2)

    hulls, simplex_src_id = \
        get_simplex(cartoon, source_cartoon_lm, delaunay_pts)
    np.save(os.path.join(output_dir, f"{DEMO_CH}_simplex"), simplex_src_id)
    np.save(os.path.join(output_dir, f"{DEMO_CH}_hulls"), hulls)
    # simplex_src_id = \
    #     np.load(os.path.join(output_dir, f"{DEMO_CH}_simplex.npy"), allow_pickle=True)
    # hulls = np.load(os.path.join(output_dir, f"{DEMO_CH}_hulls.npy"), allow_pickle=True)

    APPROACH = opt_parser.approach
    use_bg = True

    k = 0
    #  APPROACH-1 (dont use the given delaunay triangulation points)
    if APPROACH == "skimage":
        for w in warped_pts:
            dst_cartoon_lm = w.reshape(n_landmarks, 2)
            tmp = face_warp(cartoon,
                            source_cartoon_lm,
                            dst_cartoon_lm,
                            src_bg,
                            warp_only=False,
                            use_bg=use_bg,
                            )
            tmp_name = f"skimage_{DEMO_CH}_{k}.png"
            cv2.imwrite(tmp_name, tmp)
            k += 1
            if k == 3:
                # used to visualize output
                break
    # "custom" approach where you use a Piecewise affine transform based on
    # the given triangulation
    elif APPROACH == "custom":
        for w in warped_pts:
            dst_cartoon_lm = w.reshape(n_landmarks, 2)
            trans = PiecewiseAffineTransformTriang(hulls,
                                                   simplex_src_id,
                                                   reshape=True)
            trans.estimate(source_cartoon_lm, dst_cartoon_lm, delaunay_pts)
            tmp = warp(cartoon, trans)
            if use_bg:
                warped = _merge_images(tmp, src_bg)
            else:
                warped = _merge_images(tmp, cartoon)

            tmp_name = f"custom_{DEMO_CH}_{k}.png"
            cv2.imwrite(tmp_name, warped)
            k += 1
            if k == 3:
                # used to visualize output
                break