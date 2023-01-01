# Faster_rcnn_torchvision
只是用来记录跑通torchvision的faster_rcnn用到的代码，由于是使用torchvision的模型，你也可以将faster_rcnn替换成其他torchvision提供的模型。这里提供了训练、验证、onnx模型导出的代码，你可以使用我提供的 voc 数据集，按照使用演示的步骤来跑通整个过程。

### 环境搭建
 ```bash
git clone https://github.com/MadaoFY/Faster_rcnn_torchvision.git # clone

cd classification_pytorch

pip install -r requirements.txt  # install
```
### 数据集下载
我这里已经完成清洗的voc数据集。  
voc2012(训练集、验证集)：https://pan.baidu.com/s/1rICWiczIv_GyrYIrEj1p3Q  
提取码: 4pgx  
voc2007(测试集)：https://pan.baidu.com/s/1uZ1D53STUPmhXIaQ7Ce4hw  
提取码: vp5z

若你想获取原始未经清洗的voc  
voc：https://pjreddie.com/projects/pascal-voc-dataset-mirror/  
数据集讲解(来自b站up主：霹雳吧啦Wz)：https://www.bilibili.com/video/BV1kV411k7D8/?spm_id_from=333.999.0.0&vd_source=23508829e27bce925740f90e5cd28cf3

## 使用演示
### 训练(```train_rcnn.py```)
假设你已经下载了我提供的数据集，打开train_rcnn.py脚本确认参数后即可运行，部分参数如下。  
如果是从官方下载的voc数据集或者用其他的数据集，你需要对数据集进行清洗和划分，要生成coco标注格式的的json文件，具体的可参考我在voc2012文件夹中存放的train_ann.json文件。
```python
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
```
脚本的第16行左右数据增强的相关代码，37行为模型导入相关代码，91行为optimizer相关代码，你可以根据自己的需求进行修改。

```python
# 获取pytorch的faster_rcnn_v2模型
def get_faster_rcnn_model(num_classes, in_channels=3, pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=pretrained)
    # 设置输入通道数，大多数默认为3，rgb
    model.backbone.body.conv1.in_channels = in_channels
    # 设置分类的类别个数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
```

### 验证或检测(```val.py、detect.py```)
val.py脚本用于对训练好的模型进行指标验证，支持onnx模型，你可以使用onnx模型的进行验证，运行后返回预测指标。  
```bash
python val.py --weights ./models_save/faster_rcnn_model_7_57.726.pth --img_dir ./voc2007/JPEGImages/ --val_dir ./voc2007/test_ann.json --batch_size 1 -num_classes 21
```

detect.py脚本用于预测结果的导出，它会将你进行检测的图片绘制好目标框然后导出保存，这里你必须设置```--result_save_dir```参数以指定结果的保存位置，更多参数可以在脚本中查看。
```bash
python detect.py --weights ./models_save/faster_rcnn_model_7_57.726.pth --img_dir ./voc2007/JPEGImages/ --val_dir ./voc2007/test_ann.json --batch_size 1 -num_classes 21 --result_save_dir ./run/
```

### 导出onnx模型(```onnx_port.py```)
如果你需要onnx模型，可使用onnx_port.py脚本。
