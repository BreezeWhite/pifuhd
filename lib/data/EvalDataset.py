# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import random
from pathlib import Path

import numpy as np 
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import cv2
import torch
import json

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def crop_image(img, rect):
    x, y, w, h = rect

    left = abs(x) if x < 0 else 0
    top = abs(y) if y < 0 else 0
    right = abs(img.shape[1]-(x+w)) if x + w >= img.shape[1] else 0
    bottom = abs(img.shape[0]-(y+h)) if y + h >= img.shape[0] else 0
    
    if img.shape[2] == 4:
        color = [0, 0, 0, 0]
    else:
        color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    x = x + left
    y = y + top

    return new_img[y:(y+h),x:(x+w),:]


def gen_rect_by_mask(mask_path: Path):
    mask = np.array(Image.open(mask_path).convert('1'))
    idy, idx = np.where(mask > 0)
    x, y = np.min(idx), np.min(idy)
    w, h = np.max(idx) - x, np.max(idy) - y

    center_x, center_y = x + w // 2, y + h // 2
    radius = round(max(w, h) / 2 * 1.3)
    min_x = max(0, center_x - radius)
    min_y = max(0, center_y - radius)
    max_x = min(mask.shape[1], center_x + radius)
    max_y = min(mask.shape[0], center_y + radius)

    return min_x, min_y, max_x - min_x, max_y - min_y  # x, y, w, h


class EvalDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, projection='orthogonal'):
        self.opt = opt
        self.projection_mode = projection

        root = Path(self.opt.dataroot)

        if root.is_dir():
            img_files = []
            for ff in root.iterdir():
                if (
                    ff.suffix.lower() not in {'.png', '.jpeg', '.jpg'}
                    or ff.stem.endswith('_mask')
                ):
                    continue

                rect_path = ff.parent / (ff.stem + '_rect.txt')
                if rect_path.exists():
                    img_files.append(ff)
                    continue

                mask_path = ff.parent / (ff.stem + '_mask.png')
                if not mask_path.exists():
                    continue

                # Parse rect by mask
                x, y, w, h = gen_rect_by_mask(mask_path)
                np.savetxt(rect_path, [[x, y, w, h]])
                img_files.append(ff)

            self.img_files = sorted(img_files)
        else:
            self.img_files = [root]

        self.phase = 'val'
        self.load_size = self.opt.loadSize

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # only used in case of multi-person processing
        self.person_id = 0

    def __len__(self):
        return len(self.img_files)

    def get_n_person(self, index):
        img_path: Path = self.img_files[index]
        rect_path = img_path.parent / (img_path.stem + '_rect.txt')
        rects = np.loadtxt(rect_path, dtype=np.int32)

        return rects.shape[0] if len(rects.shape) == 2 else 1

    def get_item(self, index):
        img_path: Path = self.img_files[index]
        rect_path = img_path.parent / (img_path.stem + '_rect.txt')

        im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        # im = np.array(Image.open(img_path).convert('RGB'))
        if im.shape[2] == 4:
            im = im / 255.0
            im[:,:,:3] /= im[:,:,3:] + 1e-8
            im = im[:,:,3:] * im[:,:,:3] + 0.5 * (1.0 - im[:,:,3:])
            im = (255.0 * im).astype(np.uint8)
        h, w = im.shape[:2]
        
        intrinsic = np.identity(4)

        trans_mat = np.identity(4)

        rects = np.loadtxt(rect_path, dtype=np.int32)
        if len(rects.shape) == 1:
            rects = rects[None]
        pid = min(rects.shape[0]-1, self.person_id)

        rect = rects[pid].tolist()
        im = crop_image(im, rect)

        scale_im2ndc = 1.0 / float(w // 2)
        scale = w / rect[2]
        trans_mat *= scale
        trans_mat[3,3] = 1.0
        trans_mat[0, 3] = -scale*(rect[0] + rect[2]//2 - w//2) * scale_im2ndc
        trans_mat[1, 3] = scale*(rect[1] + rect[3]//2 - h//2) * scale_im2ndc
        
        intrinsic = np.matmul(trans_mat, intrinsic)
        im_512 = cv2.resize(im, (512, 512))
        im = cv2.resize(im, (self.load_size, self.load_size))

        image_512 = Image.fromarray(im_512[:,:,::-1]).convert('RGB')
        image = Image.fromarray(im[:,:,::-1]).convert('RGB')
        
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()

        calib_world = torch.Tensor(intrinsic).float()

        # image
        image_512 = self.to_tensor(image_512)
        image = self.to_tensor(image)
        return {
            'name': img_path.stem,
            'img': image.unsqueeze(0),
            'ori_img': im,
            'img_512': image_512.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'calib_world': calib_world.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def __getitem__(self, index):
        return self.get_item(index)
