import numpy as np
from sporco.signal import tikhonov_filter
import scipy
import torch
from torchvision.models.vgg import vgg19
import cv2
import argparse
import time

def lowpass(s, lda, npad):
    return tikhonov_filter(s, lda, npad)


def c3(s):
    if s.ndim == 2:
        s3 = np.dstack([s, s, s])
    else:
        s3 = s
    return np.rollaxis(s3, 2, 0)[None, :, :, :]


def l1_features(out):
    h, w, d = out.shape
    A_temp = np.zeros((h + 2, w + 2))

    l1_norm = np.sum(np.abs(out), axis=2)
    A_temp[1:h + 1, 1:w + 1] = l1_norm
    return A_temp


def fusion_strategy(feat_a, feat_b, source_a, source_b, unit):
    m, n = feat_a.shape
    m1, n1 = source_a.shape[:2]
    weight_ave_temp1 = np.zeros((m1, n1))
    weight_ave_temp2 = np.zeros((m1, n1))

    for i in range(1, m):
        for j in range(1, n):
            A1 = feat_a[i - 1:i + 1, j - 1:j + 1].sum() / 9
            A2 = feat_b[i - 1:i + 1, j - 1:j + 1].sum() / 9

            weight_ave_temp1[(i - 2) * unit + 1:(i - 1) * unit + 1, (j - 2) * unit + 1:(j - 1) * unit + 1] = A1 / (
                        A1 + A2)
            weight_ave_temp2[(i - 2) * unit + 1:(i - 1) * unit + 1, (j - 2) * unit + 1:(j - 1) * unit + 1] = A2 / (
                        A1 + A2)

    if source_a.ndim == 3:
        weight_ave_temp1 = weight_ave_temp1[:, :, None]
    source_a_fuse = source_a * weight_ave_temp1
    if source_b.ndim == 3:
        weight_ave_temp2 = weight_ave_temp2[:, :, None]
    source_b_fuse = source_b * weight_ave_temp2

    if source_a.ndim == 3 or source_b.ndim == 3:
        gen = np.atleast_3d(source_a_fuse) + np.atleast_3d(source_b_fuse)
    else:
        gen = source_a_fuse + source_b_fuse

    return gen


def get_activation(model, layer_numbers, input_image):
    outs = []
    out = input_image
    for i in range(max(layer_numbers) + 1):
        with torch.no_grad():
            out = model.features[i](out)
        if i in layer_numbers:
            outs.append(np.rollaxis(out.detach().cpu().numpy()[0], 0, 3))
    return outs


def preprocess(img_ir, img_vis):
    '''

    :param img_ir: a numpy array representing the infrared image
    :param img_vis: a numpy array representing the visible image
    :param isPi: to distinguish whether running on pc or raspberry pi
    :return: two numpy array in the same size representing the IR and VIS image
    (the size will decided by infrared image)
    '''
    # img_vis: (480, 640, 3) <class 'numpy.ndarray'>
    # img_ir: ()

    # img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
    # img_vis_cropped = img_vis[60 + 20:420 + 20, 80 - 20:560 - 20, :] # (360, 480, 3) <class 'numpy.ndarray'>
    # img_vis_resized = cv2.resize(img_vis_cropped, (320, 250), interpolation=cv2.INTER_NEAREST)
    # img_vis_resized = img_vis_resized[:240][...][...]  # (240, 320, 3)
    # img_vis_resized = cv2.cvtColor(img_vis_resized, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img_vis_preprocessed', img_vis_resized)
    # cv2.imshow('img_ir_preprocessed', img_ir)
    # return img_ir, img_vis_resized
    return img_ir, img_vis


def DLF(ir, vis, model=None):
    vis = cv2.imread(vis)
    ir = cv2.imread(ir)
    x = time.time()
    ir, vis = preprocess(ir, vis)
    npad = 16
    lda = 5
    vis_low, vis_high = lowpass(vis.astype(np.float32) / 255, lda, npad)
    ir_low, ir_high = lowpass(ir.astype(np.float32) / 255, lda, npad)
    print(vis_low.ndim, ir_low.ndim)
    if model is None:
        model = vgg19(True)
    model.cuda().eval()
    relus = [2, 7, 12, 21]
    unit_relus = [1, 2, 4, 8]

    vis_in = torch.from_numpy(c3(vis_high)).cuda()
    ir_in = torch.from_numpy(c3(ir_high)).cuda()

    relus_vis = get_activation(model, relus, vis_in)
    relus_ir = get_activation(model, relus, ir_in)

    vis_feats = [l1_features(out) for out in relus_vis]
    ir_feats = [l1_features(out) for out in relus_ir]

    saliencies = []
    saliency_max = None
    for idx in range(len(relus)):
        saliency_current = fusion_strategy(vis_feats[idx], ir_feats[idx], vis_high, ir_high, unit_relus[idx])
        saliencies.append(saliency_current)

        if saliency_max is None:
            saliency_max = saliency_current
        else:
            saliency_max = np.maximum(saliency_max, saliency_current)

    if vis_low.ndim == 3 or ir_low.ndim == 3:
        low_fused = 0.01 * np.atleast_3d(vis_low) + np.atleast_3d(ir_low)
    else:
        low_fused = 0.05 * vis_low + ir_low
    low_fused = low_fused / 2
    high_fused = saliency_max
    fused_img = low_fused + high_fused
    cv2.imshow('fused image', fused_img)
    fused_img = cv2.normalize(fused_img, None, 255., 0., cv2.NORM_MINMAX)
    fused_img = cv2.convertScaleAbs(fused_img)
    # cv2.imshow('fused image2', fused_img)
    print(time.time() - x)
    cv2.waitKey(0)
    cv2.imwrite("fused_image_dlf.jpg", fused_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--IR", default='C:/Users/svideo/Desktop/computer_ir.png', help="path to IR image", required=False)
    parser.add_argument("--VIS", default='C:/Users/svideo/Desktop/computer_vis.png', help="path to VIS image", required=False)
    a = parser.parse_args()
    DLF(a.IR, a.VIS)

