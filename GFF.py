import numpy as np
import cv2
import argparse
import logging
from layers.bicubic_layer import imresize
import time

R_G = 5
D_G = 5
bic_scale_ir = 3.4
bic_scale_vis = 0.125
rotate_angle = 15
resize_scale = 1.2


def guidedFilter(img_i, img_p, r, eps):
    # D(X)=E(X^2)-[E(X)]^2
    wsize = int(2 * r) + 1
    meanI = cv2.boxFilter(img_i, ksize=(wsize, wsize), ddepth=-1, normalize=True)  # Outputs keeps the same size with inputs
    meanP = cv2.boxFilter(img_p, ksize=(wsize, wsize), ddepth=-1, normalize=True)  # Fill with nearest neighbor
    corrI = cv2.boxFilter(img_i * img_i, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    corrIP = cv2.boxFilter(img_i * img_p, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    varI = corrI - meanI * meanI
    covIP = corrIP - meanI * meanP
    a = covIP / (varI + eps)
    b = meanP - a * meanI
    meanA = cv2.boxFilter(a, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    meanB = cv2.boxFilter(b, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    q = meanA * img_i + meanB
    return q


def GFF_GRAY(img_r, img_v):
    img_r = img_r * 1. / 255
    img_v = img_v * 1. / 255
    img_r_blur = cv2.blur(img_r, (31, 31))
    img_v_blur = cv2.blur(img_v, (31, 31))
    # cv2.imshow('img_base_layer1', img_v_blur)
    # cv2.imshow('img_base_layer2', img_r_blur)
    img_r_detail = img_r.astype(np.float64) - img_r_blur.astype(np.float64)
    img_v_detail = img_v.astype(np.float64) - img_v_blur.astype(np.float64)
    # cv2.imshow('img_detail_layer1', img_v_detail)
    # cv2.imshow('img_detail_layer2', img_r_detail)
    img_r_lap = cv2.Laplacian(img_r.astype(np.float64), -1, ksize=3)
    img_v_lap = cv2.Laplacian(img_v.astype(np.float64), -1, ksize=3)
    win_size = 2 * R_G + 1
    s1 = cv2.GaussianBlur(np.abs(img_r_lap), (win_size, win_size), R_G)
    s2 = cv2.GaussianBlur(np.abs(img_v_lap), (win_size, win_size), R_G)
    # cv2.imshow('img_saliency_map1', s2)
    # cv2.imshow('img_saliency_map2', s1)
    p1 = np.zeros_like(img_r)
    p2 = np.zeros_like(img_r)
    p1[s1 > s2] = 1
    p2[s1 <= s2] = 1
    # cv2.imshow('img_weight_map1', p2)
    # cv2.imshow('img_weight_map2', p1)
    w1_b = guidedFilter(p1, img_r.astype(np.float64), 45, 0.3)
    w2_b = guidedFilter(p2, img_v.astype(np.float64), 45, 0.3)
    w1_d = guidedFilter(p1, img_r.astype(np.float64), 7, 0.000001)
    w2_d = guidedFilter(p2, img_v.astype(np.float64), 7, 0.000001)
    # cv2.imshow('img_refined_weight_mapB1', w2_b)
    # cv2.imshow('img_refined_weight_mapB2', w1_b)
    # cv2.imshow('img_refined_weight_mapD1', w2_d)
    # cv2.imshow('img_refined_weight_mapD2', w1_d)
    w1_b_w = w1_b / (w1_b + w2_b)
    w2_b_w = w2_b / (w1_b + w2_b)
    w1_d_w = w1_d / (w1_d + w2_d)
    w2_d_w = w2_d / (w1_d + w2_d)
    fused_b = w1_b_w * img_r_blur + w2_b_w * img_v_blur
    fused_d = w1_d_w * img_r_detail + w2_d_w * img_v_detail
    # cv2.imshow('img_fused_base', fused_b)
    # cv2.imshow('img_fused_detail', fused_d)
    img_fused = fused_b + fused_d
    img_fused = cv2.normalize(img_fused, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(img_fused)


def GFF_RGB(img_r, img_v):
    fused_img = np.ones_like(img_r)
    r_R = img_r[:, :, 2]
    v_R = img_v[:, :, 2]
    r_G = img_r[:, :, 1]
    v_G = img_v[:, :, 1]
    r_B = img_r[:, :, 0]
    v_B = img_v[:, :, 0]
    fused_R = GFF_GRAY(r_R, v_R)
    fused_G = GFF_GRAY(r_G, v_G)
    fused_B = GFF_GRAY(r_B, v_B)
    fused_img[:, :, 2] = fused_R
    fused_img[:, :, 1] = fused_G
    fused_img[:, :, 0] = fused_B
    return fused_img


def preprocess(img_ir, img_vis, gray):
    """
    A preprocess for kinds of images that taken from a FLIR camera

    :param img_ir: a numpy array representing the infrared image
    :param img_vis: a numpy array representing the visible image
    :param gray: to determine whether output result should in gray or RGB
    :return: two numpy array in the same size representing the IR and VIS image
    (the size will decided by infrared image)
    """

    # img_vis: (480, 640, 3) <class 'numpy.ndarray'>
    # img_ir: ()
    logging.info('Start preprocessing the image.')
    if gray:
        img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)

    img_vis_cropped = img_vis[80:440, 60:540, ...] # (360, 480, 3) <class 'numpy.ndarray'>
    img_vis_resized = cv2.resize(img_vis_cropped, (320, 250), interpolation=cv2.INTER_NEAREST)
    img_vis_resized = img_vis_resized[:240][...][...]  # (240, 320, 3)
    cv2.imshow('img_vis_preprocessed', img_vis_resized)
    cv2.imshow('img_ir_preprocessed', img_ir)
    cv2.imwrite('img_vis_preprocessed.png', img_vis_resized)
    cv2.imwrite('img_ir_preprocessed.png', img_ir)
    return img_ir, img_vis_resized


def preprocess_pi(img_ir, img_vis, gray):
    """
    A preprocess for kinds of images that taken from a raspberry pi integrated camera

    :param img_ir: a numpy array representing the infrared image
    :param img_vis: a numpy array representing the visible image
    :param gray: to determine whether output result should in gray or RGB
    :return: two numpy array in the same size representing the IR and VIS image
    (the size will decided by infrared image)
    """

    # img_vis: (2464, 3280, 3) <class 'numpy.ndarray'>
    # img_ir: (160, 160, 3)
    # print(img_ir.shape, img_vis.shape)
    logging.info('Start preprocessing the image.')
    if gray:
        img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)

    img_ir_cropped = img_ir[35:125, 20:140, ...]
    img_vis_cropped = img_vis[2:2462, ...]
    # cv2.imshow('img_ir_cropped', img_ir_cropped)
    # cv2.imshow('img_vis_cropped', img_vis_cropped)
    # cv2.waitKey(0)

    img_ir_scaled = imresize(img_ir_cropped, bic_scale_ir, 'bicubic')
    img_vis_scaled = imresize(img_vis_cropped, bic_scale_vis, 'bicubic')
    # cv2.imshow('img_ir_scaled', img_ir_scaled)
    # cv2.imshow('img_vis_scaled', img_vis_scaled)
    # print(img_ir_scaled.shape, img_vis_scaled.shape)
    # img_vis_scaled: (308, 410, 3)
    # img_ir_scaled: (306, 408, 3)

    output_size = np.array(np.array(img_ir_scaled.shape[:2]).astype(np.float64) * resize_scale).astype(np.int32)
    img_ir_resized = cv2.resize(img_ir_scaled, output_size[::-1], interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('3.png', img_ir_resized)  # img_ir_resized: (367, 489, 3)

    # height: 367 - 308 = 59
    # width: 489 - 410 = 79
    img_ir_dst = img_ir_resized[50:358, 38:448, ...]  # (308, 410, 3)
    cv2.imwrite('img_ir_preprocessed.png', img_ir_dst)
    cv2.imwrite('img_vis_preprocessed.png', img_vis_scaled)
    cv2.imshow('img_ir_preprocessed', img_ir_dst)
    cv2.imshow('img_vis_preprocessed', img_vis_scaled)

    return img_ir_dst, img_vis_scaled


def GFF(configs):
    x = time.time()
    logging.info('Loading ir and vis image.')
    img_r = cv2.imread(configs.infrared)
    img_v = cv2.imread(configs.visible)
    if not isinstance(img_r, np.ndarray):
        logging.error(f'Can not find img_ir by path {configs.infrared}')
        return
    if not isinstance(img_v, np.ndarray):
        logging.error(f'Can not find img_vis by path {configs.visible}')
        return
    logging.info('Image loading complete.')
    if configs.isRaspberryPi:
        img_r, img_v = preprocess_pi(img_r, img_v, configs.isGray)
    else:
        img_r, img_v = preprocess(img_r, img_v, configs.isGray)
    logging.info('Image preprocessing is complete.')
    if img_r.shape[0] != img_v.shape[0] or img_r.shape[1] != img_v.shape[1]:
        logging.error('The sizes of those two images do not match.')
        return
    fused_img = None
    logging.info('Starting fuse two images based on guided filtering.')
    if len(img_r.shape) < 3 or img_r.shape[2] == 1:
        if len(img_v.shape) < 3 or img_v.shape[-1] == 1:
            fused_img = GFF_GRAY(img_r, img_v)
        else:
            img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
            fused_img = GFF_GRAY(img_r, img_v_gray)
    else:
        if len(img_v.shape) < 3 or img_v.shape[-1] == 1:
            img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = GFF_GRAY(img_r_gray, img_v)
        else:
            fused_img = GFF_RGB(img_r, img_v)
    # fused_img = cv2.resize(fused_img, (352, 352), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('fused image', fused_img)
    cv2.imwrite("fused_image_gff.jpg", fused_img)
    print(time.time() - x)
    cv2.waitKey(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--infrared', type=str,
    #                     default='C:\\Users\\svideo\\Desktop\\2021 A semester\\Year2 Project\\dataset\\IR\\TNO_Image_Fusion_Dataset\\Athena_images\\2_men_in_front_of_house\\IR_meting003_g.bmp',
    #                     help='input IR image path', required=False)
    # parser.add_argument('-v', '--visible', type=str,
    #                     default='C:\\Users\\svideo\\Desktop\\2021 A semester\\Year2 Project\\dataset\\IR\\TNO_Image_Fusion_Dataset\\Athena_images\\2_men_in_front_of_house\\VIS_meting003_r.bmp',
    #                     help='input Visible image path', required=False)

    parser.add_argument('-r', '--infrared', type=str,
                        default='C:/Users/svideo/Desktop/2021 A semester/Year2 Project Material/demonstration/FLIR/FLIR0065.jpg',
                        help='input IR image path', required=False)
    parser.add_argument('-v', '--visible', type=str,
                        default='C:/Users/svideo/Desktop/2021 A semester/Year2 Project Material/demonstration/FLIR/FLIR0066.jpg',
                        help='input Visible image path', required=False)
    parser.add_argument('-pi', '--isRaspberryPi', action='store_true', default=0,
                        help='to distinguish whether running on pc or raspberry pi')

    # parser.add_argument('-r', '--infrared', type=str,
    #                     default='C:\\Users\\svideo\\Desktop\\thermal visible pairs\\thermal_camera_image\\thermal_2022-02-24_21_03_12.86.png',
    #                     help='input IR image path', required=False)
    # parser.add_argument('-v', '--visible', type=str,
    #                     default='C:\\Users\\svideo\\Desktop\\thermal visible pairs\\visible_camera_image\\visible_2022-02-24_21_03_12.87.jpg',
    #                     help='input Visible image path', required=False)
    # parser.add_argument('-pi', '--isRaspberryPi', action='store_true', default=1,
    #                     help='to distinguish whether running on pc or raspberry pi')
    parser.add_argument('-g', '--isGray', action='store_true', default=1,
                        help='to determine whether output fusion result in gray or RGB')
    args = parser.parse_args()

    logging.info('Initialization finished.')
    GFF(args)

