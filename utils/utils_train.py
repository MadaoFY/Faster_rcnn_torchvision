import torch
import cv2 as cv
import numpy as np

from tqdm.auto import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

__all__ = [
    'train_detection',
    'outs_to_coco'
]


def outs_to_coco(img_ids, predictions, threshold=0.0):
    results = []
    for img_id, one_pre in zip(img_ids, predictions):
        # 需要固定batch size 为 1，否则出错
        data_dict = one_pre
        if 'scores' not in data_dict:
            continue

        scores = data_dict['scores']
        mask = scores > threshold
        scores = scores[mask]
        if sum(scores) == 0:
            continue

        labels = data_dict['labels'][mask]
        boxes = data_dict['boxes'][mask]
        boxes[:, 2:] -= boxes[:, :2]
        pre_num = len(scores)
        for i in range(pre_num):
            score = float(scores[i])
            label_id = int(labels[i])
            bbox = boxes[i]
            bbox = bbox.tolist()
            out = {
                'image_id': int(img_id),
                'category_id': label_id,
                'bbox': bbox,
                'score': score
            }
            results.append(out)

    return results


def train_detection(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        epochs,
        model_save_dir,
        log_save_dir,
        model_save_epochs=None,
        device='cuda',
        fp16=True
):
    if log_save_dir is not None:
        writer = SummaryWriter(log_save_dir)

    best_map50 = 0.0
    coco_true = COCO(annotation_file=val_loader.dataset.ann_dir)
    device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    if fp16:
        scaler = GradScaler()

    for epoch in range(epochs):
        epoch += 1
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()
        # These are used to record information in training.
        loss_keys = None
        train_loss = torch.tensor([], device=device)

        # Iterate the training set by batches.
        for images, targets in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if fp16:
                with autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                # Compute the gradients for parameters.
                # 反向传播
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                # Gradients stored in the parameters in the previous step should be cleared out first.
                # 梯度清零
                optimizer.zero_grad()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Compute the gradients for parameters.
                losses.backward()
                optimizer.step()
                # Gradients stored in the parameters in the previous step should be cleared out first.
                # 梯度清零
                optimizer.zero_grad()

            # Record the loss.
            train_loss = torch.concatenate((train_loss, losses.detach().unsqueeze(0)))

            if loss_keys is None:
                loss_keys = list(loss_dict.keys())
                loss_values = torch.zeros(len(loss_keys), device=device).unsqueeze(0)
            loss_value = torch.tensor(list(loss_dict.values()), device=device).detach().unsqueeze(0)
            loss_values = torch.cat((loss_values, loss_value), dim=0)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = train_loss.mean().cpu().numpy()
        loss_values = loss_values[1:, :].mean(0).cpu().numpy()

        # Print the information.
        # 打印每轮的loss信息
        print(
            f"[Train | {epoch:03d}/{epochs:03d} ] lr={optimizer.state_dict()['param_groups'][0]['lr']:.5f}, "
            f"loss = {train_loss:.5f}"
        )
        l = []
        for k, v in zip(loss_keys, loss_values):
            l.append(f"{k}={v:.5f}")
        print(', '.join(l))

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        predictions = []
        id_list = []

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
            image_id = [t['image_id'].cpu().numpy() for t in targets]

            predictions.extend(outputs)
            id_list.extend(image_id)

        res = outs_to_coco(id_list, predictions)

        coco_pre = coco_true.loadRes(res)
        coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        val_map50 = coco_evaluator.stats[0] * 100

        if log_save_dir is not None:
            writer.add_scalar('loss_total', train_loss, global_step=epoch)
            writer.add_scalars('loss_reg_cls',
                               {k: v for k, v in zip(loss_keys, loss_values)},
                               global_step=epoch)

            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
            writer.add_scalar('map0.5:0.95', val_map50, global_step=epoch)

        # Print the information.
        print(f"[Valid | {epoch:03d}/{epochs:03d} ] map0.5:0.95 = {val_map50:.3f}")

        # if the model improves, save a checkpoint at this epoch
        # 保存训练权值
        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save(model.state_dict(), model_save_dir + f'_{epoch}_{best_map50:.3f}.pth')
            print('{}[ saving model with val map0.5:0.95 {:.3f} ]{}'.format('-' * 15, best_map50, '-' * 15))
        if epoch in model_save_epochs:
            torch.save(model.state_dict(), model_save_dir + f'_{epoch}_{val_map50:.3f}.pth')
            print(f'saving model with epoch {epoch}')
    if log_save_dir is not None:
        writer.close()
    print(f'Done!!!best map0.5:0.95 = {best_map50:.3f}')
