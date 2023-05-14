import os
import time
import torch
import torchvision
import albumentations as A

from albumentations import pytorch as AT
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils.dataset import ReadDataSet, collate_fn
from utils.general import SineAnnealingLR, same_seeds
from utils.utils_train import train_detection


# 数据增强操作
def get_transform(train=True):
    transforms = []
    if train:
        transforms.append(A.OneOf([
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=1),
            A.RandomBrightnessContrast(p=1)
        ], p=0.5))
        # transforms.append(A.OneOf([
        #     A.Emboss(p=1),
        #     A.Sharpen(p=1)
        # ], p=0.3))
        transforms.append(A.OneOf([
            A.GaussianBlur(p=1),
            A.MedianBlur(p=1)
        ], p=0.3))
        transforms.append(A.HorizontalFlip(p=0.5))
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
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    same_seeds(42)

    # 读取训练集验证集
    img_dir = os.path.join(args.img_dir)

    train_dataset = ReadDataSet(args.train_dir, img_dir, get_transform(), False)
    val_dataset = ReadDataSet(args.valid_dir, img_dir, get_transform(train=False), False)

    # 设置batch大小
    batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 学习率
    lr = args.lr
    weight_decay = args.weight_decay
    # 训练轮次
    epochs = args.epochs
    # 模型权重保存路径
    model_save_dir = args.model_save_dir
    # 创建模型
    model = get_faster_rcnn_model(num_classes=args.num_classes, pretrained=args.pretrain_coco).to(device)
    # for i, p in enumerate(model.parameters()):
    #     if i < 1:
    #         p.requires_grad = False

    # 创建优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
        betas=(0.9, 0.999), weight_decay=weight_decay
    )
    # 优化策略cosine
    t_max = 20
    lr_cosine = SineAnnealingLR(optimizer, t_max)
    # lr_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    # 是否使用半精度训练
    fp16 = args.fp16
    # 用于计算训练时间
    start = time.time()

    # 模型训练
    train_detection(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_cosine,
        epochs,
        model_save_dir,
        log_save_dir=args.log_save_dir,
        model_save_epochs=args.model_save_epochs,
        device=args.device,
        fp16=fp16
    )
    print(f'{epochs} epochs completed in {(time.time() - start) / 3600.:.3f} hours.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='训练设备类型')
    # 训练所需图片的根目录
    parser.add_argument('--img_dir', default='./voc2012/JPEGImages/', help='训练所用图片根目录')
    # 训练集
    parser.add_argument('--train_dir', default='./voc2012/train_ann.json', help='训练集文档')
    # 验证集
    parser.add_argument('--valid_dir', default='./voc2012/val_ann.json', help='测试集文档')
    # 载入coco预训练权重
    parser.add_argument('--pretrain_coco', default=True, help='预训练权重')
    # 训练信息保存位置
    parser.add_argument('--log_save_dir', default=None, help='tensorboard信息保存地址')
    # 模型权重保存位置
    parser.add_argument('--model_save_dir', default='./models_save/faster_rcnn_model',
                        help='模型权重保存位置')
    # 学习率
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate, 0.0001 is the default value for training')
    # 训练的总epoch数
    parser.add_argument('--epochs', type=int, default=330, metavar='N', help='number of total epochs to run')
    # 优化器的weight_decay参数
    parser.add_argument('--weight-decay', type=float, default=0.1, metavar='W', help='weight decay')
    # 训练的batch_size
    parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='batch size when training')
    # 目标分类数量
    parser.add_argument('--num_classes', type=int, default=21, help='数据集分类数量')
    # 额外指定权重保存epoch
    parser.add_argument('--model_save_epochs', type=list, default=[39,40,41,42,43], metavar='N',
                        help='额外指定epoch进行权重保存')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--fp16", default=True, choices=[True, False], help="Use fp16 for mixed precision training")

    args = parser.parse_args()
    print(args)

    main(args)
