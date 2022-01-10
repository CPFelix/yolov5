import cv2
import numpy as np

INPUT_HEIGHT = 640
INPUT_WIDTH = 640

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def preprocess(image):
    image, ratio, (dw, dh) = letterbox(image)
    # Convert
    img = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img /= 255.0
    img = np.ascontiguousarray(img)
    return img, ratio, (dw, dh)

def nms(boxes, box_confidences, nms_threshold=0.5):
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            iou = intersection / union

            indexes = np.where(iou <= nms_threshold)[0]
            ordered = ordered[indexes + 1]
        keep = np.array(keep).astype(int)
        return keep

def postprocess(buffer, ratio, dw, dh, conf_threshold=0.6, nms_threshold=0.5):
    detected_objects = []

    num_bboxes = int(buffer[0, 0, 0, 0])

    if num_bboxes:
        #bboxes = buffer[0, 1 : (num_bboxes * 7 + 1), 0, 0].reshape(-1, 7)
        # Reshape to a two dimentional ndarray
        bboxes = np.reshape(buffer[0, 1:, 0, 0 ], (-1, 6))[:num_bboxes, :]
        labels = set(bboxes[:, 5].astype(int))
        for label in labels:
            selected_bboxes = bboxes[np.where((bboxes[:, 5] == label) & (bboxes[:, 4] >= conf_threshold))]
            selected_bboxes_keep = selected_bboxes[nms(selected_bboxes[:, :4], selected_bboxes[:, 4], nms_threshold)]
            for idx in range(selected_bboxes_keep.shape[0]):
                box_xy = selected_bboxes_keep[idx, :2] - [dw, dh]
                box_wh = selected_bboxes_keep[idx, 2:4]
                score = selected_bboxes_keep[idx, 4]

                box_x1y1 = box_xy - (box_wh / 2)
                box_x2y2 = np.minimum(box_xy + (box_wh / 2), [INPUT_WIDTH, INPUT_HEIGHT])
                box = np.concatenate([box_x1y1, box_x2y2])
                box /= ratio[0]

                if box[0] == box[2]:
                    continue
                if box[1] == box[3]:
                    continue
                detected_objects.append([label, score, box[0], box[1], box[2], box[3]])
    return detected_objects