import os
import torch
import onnxruntime
import torchvision
import numpy as np
import pandas as pd
import albumentations as A
from tqdm.auto import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from albumentations import pytorch as AT
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils.dataset import ReadDataSet, collate_fn
from utils.utils_train import outs_to_coco
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def val_transform():
    transforms = []
    transforms.append(AT.ToTensorV2())
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


# 获取pytorch的faster_rcnn_v2模型
def get_faster_rcnn_model(num_classes, in_channels=3, pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=pretrained)
    # 设置输入通道数，大多数默认为3，rgb
    model.backbone.body.conv1.in_channels = in_channels
    # 设置分类的类别个数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main(args):
    img_dir = args.img_dir
    val_dir = args.val_dir

    val_dataset = ReadDataSet(val_dir, img_dir, val_transform(), False)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
        )

    coco_true = COCO(annotation_file=val_loader.dataset.ann_dir)

    weights = args.weights
    last_name = os.path.splitext(weights)[-1]
    assert last_name in ['.pth', '.pt', '.onnx'], f"weights file attribute is {last_name}, not in [.pth , .pt, .onnx]."

    predictions = []
    id_list = []
    if last_name == '.onnx':
        model = onnxruntime.InferenceSession(
            weights,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )


        keys = ['boxes', 'labels', 'scores']
        for images, targets in tqdm(val_loader):
            images = np.asarray([i.float().numpy() for i in images])
            inputs = {model.get_inputs()[0].name: images}
            outputs = model.run(None, inputs)

            outputs = [{k: v for k, v in zip(keys, outputs)}]
            outputs = [{k: v for k, v in t.items()} for t in outputs]
            images_id = [t['image_id'].numpy() for t in targets]

            predictions.extend(outputs)
            id_list.extend(images_id)

    elif last_name in ['.pth', '.pt']:

        num_classes = args.num_classes
        model = get_faster_rcnn_model(num_classes=num_classes, pretrained=True).to(device)  #257
        param_weights = torch.load(weights)
        model.load_state_dict(param_weights, strict=True)

        model.eval()

        # Iterate the validation set by batches.
        for images, targets in tqdm(val_loader):
            images = list(img.float().to(device) for img in images)
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            # if fp16:
            #     with autocast():
            with torch.no_grad():
                outputs = model(images)

            outputs = [{k: v.cpu().numpy() for k, v in t.items()} for t in outputs]
            images_id = [t['image_id'].cpu().numpy() for t in targets]

            predictions.extend(outputs)
            id_list.extend(images_id)

    else:
        pass

    res = outs_to_coco(id_list, predictions)

    coco_pre = coco_true.loadRes(res)
    coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    val_map50 = coco_evaluator.stats[0] * 100

    print(f"[Val | map0.5:0.95 = {val_map50:.3f}]")
    print("Done!!!!!!!!!!!!!!!!!!!!")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 权重
    parser.add_argument('--weights', default='./models_save/faster_rcnn_model_7_57.726.pth',
                        help='模型文件地址; pth,pt,onnx模型')
    # 推理所需图片的根目录
    parser.add_argument('--img_dir', default='./voc2007/JPEGImages/', help='训练所用图片根目录')
    # 验证集
    parser.add_argument('--val_dir', default='./voc2007/test_ann.json', help='验证集文档')
    # batch_size
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size when training')
    # 数据集分类数量
    parser.add_argument('--num_classes', type=int, default=21, help='数据集分类数量')

    args = parser.parse_args()
    print(args)

    main(args)