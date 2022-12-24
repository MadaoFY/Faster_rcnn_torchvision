import json
import os

import torch
import cv2 as cv
import numpy as np
from os import path
from torch.utils.data import Dataset

from utils.utils_detection import letterbox_image

__all__ = [
    'ReadDataSet',
    'collate_fn'
]


class ReadDataSet(Dataset):
    r"""
    参数说明：
        ann_dir：以coco格式记录图片标注框信息的json文件地址,如果只是用于检测输出,可设置为None，将默认以img_dir文件里的图片为测试集。
        img_dir：图片所在的文件夹路径。
        letterbox(bool)：是否把短边填充到与长边一样的长度。
        transform：图片增强方法。
        test(bool)：是否为测试集。默认值：False
    """

    def __init__(self, ann_dir, img_dir, transforms=None, letterbox=False, test=False):
        super().__init__()
        self.ann_dir = ann_dir
        self.img_dir = img_dir
        self.letterbox = letterbox
        self.transforms = transforms
        self.test = test

        if self.test and self.ann_dir is None:
            # self.img_list = os.listdir(self.img_dir)
            self.img_list = [{'file_name': v} for v in os.listdir(self.img_dir)]
        else:
            with open(self.ann_dir, "r") as f:
                data_json = json.load(f)
            self.img_list = data_json['images']

            if data_json.get('annotations'):
                self.ann_list = []
                for i in data_json['annotations']:
                    self.ann_list.append(list(i.values()))
                self.ann_list = np.asarray(self.ann_list, dtype=object)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_name = self.img_list[idx]["file_name"]
        # 读取图片和label
        img = cv.imread(path.join(self.img_dir, image_name))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if self.letterbox:
            img, p = letterbox_image(img, return_padding=True)

        if not self.test:
            image_id = self.img_list[idx]['id']
            img_ann = self.ann_list[self.ann_list[:, 1] == image_id]
            bboxes = np.array([])
            for i in img_ann[:, 3]:
                bboxes = np.concatenate((bboxes, np.asarray(i)))

            labels = img_ann[:, 4]
            bboxes = bboxes.reshape(-1, 4)
            areas = bboxes[:, 2] * bboxes[:, 3]
            bboxes[:, 2:] += bboxes[:, :2]
            if self.letterbox:
                if h > w:
                    bboxes[:, [0, 2]] += p
                else:
                    bboxes[:, [1, 3]] += p

            image_id = torch.as_tensor(image_id)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(img_ann[:, 5].astype('int'), dtype=torch.int8)

            target = {"image_id": image_id, "area": areas, "iscrowd": iscrowd}

            if self.transforms:
                transformed = self.transforms(image=img, bboxes=bboxes, labels=labels)
                img = transformed["image"]
                bboxes = transformed["bboxes"]
                labels = transformed["labels"]

            target['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

            return img, target

        else:
            if self.transforms:
                transformed = self.transforms(image=img)
                img = transformed["image"]

            return img, image_name


def collate_fn(batch):
    return tuple(zip(*batch))