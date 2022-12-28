import os
import json
import torch
import cv2 as cv
import onnxruntime
import torchvision
import numpy as np
import albumentations as A
from shutil import copy
from tqdm.auto import tqdm
from albumentations import pytorch as AT
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils.dataset import ReadDataSet, collate_fn
from utils.general import yaml_load, mk_results_dirs
from utils.utils_detection import Colors, draw_boxes, non_max_suppression_fasterrcnn
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_transform():
    transforms = []
    transforms.append(AT.ToTensorV2())
    return A.Compose(transforms)


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

    test_dataset = ReadDataSet(val_dir, img_dir, test_transform(), False, test=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
        )
    # 检测物体标签
    catid_labels = yaml_load(args.labels)['labels']
    # 为每个类别分配颜色
    color = Colors(catid_labels)
    color_dicts = color.get_id_and_colors()
    # 创建保存检测图片的文件夹
    result_save_dir = mk_results_dirs(args.result_save_dir)

    weights = args.weights
    last_name = os.path.splitext(weights)[-1]
    assert last_name in ['.pth', '.pt', '.onnx'], f"weights file attribute is {last_name}, not in [.pth , .pt, .onnx]."
    # onnx推理
    if last_name == '.onnx':
        model = onnxruntime.InferenceSession(
            weights,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        for images, images_name in tqdm(test_loader):
            images = np.asarray([i.float().numpy() for i in images])
            inputs = {model.get_inputs()[0].name: images}
            outputs = model.run(None, inputs)

            for i in range(len(images_name)):
                pre = np.concatenate((outputs[0], np.expand_dims(outputs[2], 1), np.expand_dims(outputs[1], 1)), axis=1)
                pre = torch.as_tensor(pre, device=device).unsqueeze(0)
                pre = non_max_suppression_fasterrcnn(pre, conf_thres=args.conf_thres, iou_thres=args.iou_thres)[0]
                images_txt = images_name[i].split('.')[0] + '.txt'
                with open(f'{result_save_dir}/labels/{images_txt}', 'w') as f:
                    if pre is not None:
                        img0 = cv.imread(f'{args.img_dir}/{images_name[i]}')
                        pre = pre.cpu().numpy()
                        for b in pre:
                            l = b[-1] - 1
                            f.write(f'{catid_labels[l]} {b[0]} {b[1]} {b[2]} {b[3]} {b[-2]}\n')
                            img0 = draw_boxes(img0, b[:4], b[-2], l, catid_labels, 0.5, color_dicts)
                            cv.imwrite(f'{result_save_dir}/{images_name[i]}', img0)
                    else:
                        copy(f'{args.img_dir}/{images_name[i]}', f'{result_save_dir}/{images_name[i]}')

    # pytorch推理
    elif last_name in ['.pth', '.pt']:

        num_classes = args.num_classes
        model = get_faster_rcnn_model(num_classes=num_classes, pretrained=True).to(device)
        param_weights = torch.load(weights)
        model.load_state_dict(param_weights, strict=True)

        model.eval()
        # Iterate the validation set by batches.
        for images, images_name in tqdm(test_loader):
            images = list(img.float().to(device) for img in images)
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            # if fp16:
                # with autocast():
            with torch.no_grad():
                outputs = model(images)

            for i in range(len(images_name)):
                boxes = outputs[i]['boxes']
                labels = outputs[i]['labels'].unsqueeze(1)
                scores = outputs[i]['scores'].unsqueeze(1)
                pre = torch.cat((boxes, scores, labels), dim=1).unsqueeze(0)
                pre = non_max_suppression_fasterrcnn(pre, conf_thres=args.conf_thres, iou_thres=args.iou_thres)[0]
                images_txt = images_name[i].split('.')[0] + '.txt'
                with open(f'{result_save_dir}/labels/{images_txt}', 'w') as f:
                    if pre is not None:
                        img0 = cv.imread(f'{img_dir}/{images_name[i]}')
                        pre = pre.cpu().numpy()
                        for b in pre:
                            l = b[-1] - 1
                            f.write(f'{catid_labels[l]} {b[0]} {b[1]} {b[2]} {b[3]} {b[-2]}\n')
                            img0 = draw_boxes(img0, b[:4], b[-2], l, catid_labels, 0.5, color_dicts)
                            cv.imwrite(f'{result_save_dir}/{images_name[i]}', img0)
                    else:
                        copy(f'{img_dir}/{images_name[i]}', f'{result_save_dir}/{images_name[i]}')

    else:
        pass


    print("Done!!!!!!!!!!!!!!!!!!!!")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 权重
    parser.add_argument('--weights', default='./models_save/faster_rcnn_model_7_57.726.pth',
                        help='模型文件地址; pth,pt,onnx模型')
    # 目标类别标签
    parser.add_argument('--labels', type=str, default='./labels_voc.yaml', help='obj labels')
    # 推理所需图片的根目录
    parser.add_argument('--img_dir', type=str, default='./voc2007/JPEGImages/', help='训练所用图片根目录')
    # 验证集
    parser.add_argument('--val_dir', default='./voc2007/test_ann.json', help='验证集文档')
    # 检测结果保存位置
    parser.add_argument('--result_save_dir', default='./run/', help='检测结果保存的文件夹')
    # batch_size
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size when training')
    # 数据集分类数量
    parser.add_argument('--num_classes', type=int, default=21, help='数据集分类数量')
    # 只有得分大于置信度的预测框会被保留下来
    parser.add_argument('--conf_thres', type=float, default=0.45, help='confidence threshold')
    # 非极大抑制所用到的nms_iou大小
    parser.add_argument('--iou_thres', type=float, default=0.3, help='NMS IoU threshold')

    args = parser.parse_args()
    print(args)

    main(args)