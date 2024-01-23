import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import random

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


ImSurf = np.array([255, 255, 255])  # label 0
Building = np.array([255, 0, 0])  # label 1
LowVeg = np.array([255, 255, 0])  # label 2
Tree = np.array([0, 255, 0])  # label 3
Car = np.array([0, 255, 255])  # label 4
Clutter = np.array([0, 0, 255])  # label 5
Boundary = np.array([0, 0, 0])  # label 6
num_classes = 6


def get_img_mask_padded(image, mask, patch_size, mode):
    img, mask = np.array(image), np.array(mask)
    oh, ow = img.shape[0], img.shape[1]
    rh, rw = oh % patch_size, ow % patch_size
    width_pad = 0 if rw == 0 else patch_size - rw
    height_pad = 0 if rh == 0 else patch_size - rh

    h, w = oh + height_pad, ow + width_pad

    pad_img = np.pad(img, ((0, height_pad), (0, width_pad), (0, 0)), mode='constant', constant_values=0)
    pad_mask = np.pad(mask, ((0, height_pad), (0, width_pad), (0, 0)), mode='constant', constant_values=6)

    return pad_img, pad_mask


def pv2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 255, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb





def rgb_to_2D_label(_label):
    _label = _label.transpose(2, 0, 1)
    label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)
    label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0
    label_seg[np.all(_label.transpose([1, 2, 0]) == Building, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == LowVeg, axis=-1)] = 2
    label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 3
    label_seg[np.all(_label.transpose([1, 2, 0]) == Car, axis=-1)] = 4
    label_seg[np.all(_label.transpose([1, 2, 0]) == Clutter, axis=-1)] = 5
    label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] = 6
    return label_seg


def image_augment(image, mask, patch_size, mode='train', val_scale=1.0):
    image_list = []
    mask_list = []
    image_width, image_height = image.shape[1], image.shape[0]
    mask_width, mask_height = mask.shape[1], mask.shape[0]
    assert image_height == mask_height and image_width == mask_width
    if mode == 'train':
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

    else:
        image = cv2.resize(image, (int(image_width * val_scale), int(image_height * val_scale)))
        mask = cv2.resize(mask, (int(mask_width * val_scale), int(mask_height * val_scale)))

    image, mask = get_img_mask_padded(image, mask, patch_size, mode)
    mask = rgb_to_2D_label(mask)

    image_list.append(image)
    mask_list.append(mask)

    return image_list, mask_list


def car_aug(image, mask):
    assert image.shape[:2] == mask.shape
    image_list = []
    mask_list = []

    resize_crop_1 = cv2.resize(image, (int(image.shape[0] * 1.25), int(image.shape[1] * 1.25)))
    resize_crop_1 = resize_crop_1[:image.shape[0], :image.shape[1]]
    resize_crop_2 = cv2.resize(image, (int(image.shape[0] * 1.5), int(image.shape[1] * 1.5)))
    resize_crop_2 = resize_crop_2[:image.shape[0], :image.shape[1]]
    resize_crop_3 = cv2.resize(image, (int(image.shape[0] * 1.75), int(image.shape[1] * 1.75)))
    resize_crop_3 = resize_crop_3[:image.shape[0], :image.shape[1]]
    resize_crop_4 = cv2.resize(image, (int(image.shape[0] * 2.0), int(image.shape[1] * 2.0)))
    resize_crop_4 = resize_crop_4[:image.shape[0], :image.shape[1]]
    v_flip = cv2.flip(image, 0)
    h_flip = cv2.flip(image, 1)
    rotate_90 = np.rot90(image)

    image_list.extend([image, resize_crop_1,resize_crop_2, resize_crop_3, resize_crop_4, v_flip, h_flip, rotate_90])
    mask_list.extend([mask] * 8)

    return image_list, mask_list


def patch_format(inp):
    (img_path, mask_path, imgs_output_dir, masks_output_dir, eroded, gt, rgb_image,
     mode, val_scale, split_size, stride) = inp
    img_filename = os.path.basename(img_path)
    mask_filename = os.path.basename(mask_path)

    if eroded:
        mask_path = mask_path + '_label_noBoundary.tif'
    else:
        mask_path = mask_path + '_label.tif'
    if rgb_image:
        img_path = img_path + '_RGB.tif'
    else:
        img_path = img_path + '_IRRG.tif'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
   
    out_origin_mask_path = os.path.join(masks_output_dir + '/origin/', "{}.tif".format(mask_filename))
    cv2.imwrite(out_origin_mask_path, mask)
    image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), patch_size=split_size,
                                          val_scale=val_scale, mode=mode)
    assert img_filename == mask_filename and len(image_list) == len(mask_list)
    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        mask = mask_list[m]
        assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]
        if gt:
            mask = pv2rgb(mask.copy())

        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                img_tile_cut = img[y:y + split_size, x:x + split_size]
                mask_tile_cut = mask[y:y + split_size, x:x + split_size]
                img_tile, mask_tile = img_tile_cut, mask_tile_cut

                if img_tile.shape[0] == split_size and img_tile.shape[1] == split_size \
                        and mask_tile.shape[0] == split_size and mask_tile.shape[1] == split_size:

                    bins = np.array(range(num_classes + 1))
                    class_pixel_counts, _ = np.histogram(mask_tile, bins=bins)
                    cf = class_pixel_counts / (mask_tile.shape[0] * mask_tile.shape[1])
                    if cf[4] > 1.0 and mode == 'train':
                        car_imgs, car_masks = car_aug(img_tile, mask_tile)
                        for i in range(len(car_imgs)):
                            out_img_path = os.path.join(imgs_output_dir,
                                                        "{}_{}_{}_{}.tif".format(img_filename, m, k, i))
                            cv2.imwrite(out_img_path, car_imgs[i])

                            out_mask_path = os.path.join(masks_output_dir,
                                                         "{}_{}_{}_{}.png".format(mask_filename, m, k, i))
                            cv2.imwrite(out_mask_path, car_masks[i])
                    else:
                        out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.tif".format(img_filename, m, k))
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(mask_filename, m, k))
                        cv2.imwrite(out_mask_path, mask_tile)

                k += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="data/potsdam/train_images")
    parser.add_argument("--mask-dir", default="data/potsdam/train_masks")
    parser.add_argument("--output-img-dir", default="data/potsdam/train/images_1024")
    parser.add_argument("--output-mask-dir", default="data/potsdam/train/masks_1024")
    parser.add_argument("--eroded", action='store_true')
    parser.add_argument("--gt", action='store_true')  # output RGB mask
    parser.add_argument("--rgb-image", action='store_true')  # use Potsdam RGB format images
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--val-scale", type=float, default=1.0)  # ignore
    parser.add_argument("--split-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    return parser.parse_args()

if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    eroded = args.eroded
    gt = args.gt
    rgb_image = args.rgb_image
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride
    img_paths_raw = glob.glob(os.path.join(imgs_dir, "*.tif"))
    img_paths = [p[:-9] for p in img_paths_raw]
    if rgb_image:
        img_paths = [p[:-8] for p in img_paths_raw]
    mask_paths_raw = glob.glob(os.path.join(masks_dir, "*.tif"))
    if eroded:
        mask_paths = [(p[:-21]) for p in mask_paths_raw]
    else:
        mask_paths = [p[:-10] for p in mask_paths_raw]
    img_paths.sort()
    mask_paths.sort()

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        if gt:
            os.makedirs(masks_output_dir+'/origin')

    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir, eroded, gt, rgb_image,
            mode, val_scale, split_size, stride)
           for img_path, mask_path in zip(img_paths, mask_paths)]

    t0 = time.time()
    #使用线程，需要设置为8
    mpp.Pool(processes=8).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images splitting spends: {} s'.format(split_time))

