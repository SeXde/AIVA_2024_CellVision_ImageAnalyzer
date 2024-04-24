import torchvision
import torch
from functools import partial
from torchvision.models.detection.fcos import FCOSClassificationHead
import cv2
from src.train import train_constants


def create_fcos_model(num_classes=91, min_size=640, max_size=640):
    model = torchvision.models.detection.fcos_resnet50_fpn(
        weights='DEFAULT'
    )
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = FCOSClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    for param in model.parameters():
        param.requires_grad = True
    return model


def model_outputs_to_bbox(outputs, orig_image, image, score_threshold=0.6, nms_threshold=0.7):
    colors = [
        [0, 0, 0],
        [255, 0, 0],
    ]
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # Carry further only if there are detected boxes.
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # Filter out boxes according to `detection_threshold`.
        # boxes = boxes[scores >= args.threshold].astype(np.int32)
        idx = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=score_threshold, nms_threshold=nms_threshold)
        boxes = boxes[idx]
        draw_boxes = boxes.copy()
        # Get all the predicited class names.
        pred_classes = [train_constants.CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        # Draw the bounding boxes and write the class name on top of it.
        bboxes = []
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = colors[train_constants.CLASSES.index(class_name)]
            # Recale boxes.
            x_min = int((box[0] / image.shape[1]) * orig_image.shape[1])
            y_min = int((box[1] / image.shape[0]) * orig_image.shape[0])
            x_max = int((box[2] / image.shape[1]) * orig_image.shape[1])
            y_max = int((box[3] / image.shape[0]) * orig_image.shape[0])
            bboxes.append(((x_min, y_min), (x_max, y_max)))

        return bboxes


def print_bboxes(image, bboxes, print_image=True):
    for (x1, y1), (x2, y2) in bboxes:
        cv2.rectangle(image,
                      (x1, y1),
                      (x2, y2),
                      (0, 0, 255),
                      3)
        cv2.putText(image,
                    'Cell',
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    lineType=cv2.LINE_AA)
    if print_image:
        cv2.imshow('Cells', image)
        cv2.waitKey(0)
    return image
