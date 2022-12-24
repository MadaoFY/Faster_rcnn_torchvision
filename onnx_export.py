import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# 获取pytorch的faster_rcnn_v2模型
def get_faster_rcnn_model(num_classes, in_channels=3, pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=pretrained)
    # 设置输入通道数，大多数默认为3，rgb
    model.backbone.body.conv1.in_channels = in_channels
    # 设置分类的类别个数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == '__main__':

    root = '../voc/'
    model_dir = os.path.join(root, 'models_save/faster_rcnn_model_7_57.726')

    x = torch.randn(1, 3, 800, 800, requires_grad=True)
    torch_model = get_faster_rcnn_model(num_classes=21, pretrained=False)
    param = torch.load(f'{model_dir}.pth')
    torch_model.load_state_dict(param, strict = False)


    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      f'{model_dir}.onnx',   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=13,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names=['boxes', 'labels', 'scores'],  # the model's output names
                      dynamic_axes={'input' : {
                          # 0 : 'batch',
                          2 : 'height',
                          3 : 'width'
                      },    # variable length axes
                                     }
                )
