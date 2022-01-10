# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

import cv2
import torch
import torchvision
import time

from labels import COCOLabels

RAND_COLORS = np.random.randint(50, 255, (64, 3), "int")  # used for class visu


def nms(boxes, box_confidences, nms_threshold=0.5):
    x_coord = boxes[:, 0]
    y_coord = boxes[:, 1]
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]

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

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def postprecess(prediction, conf_thres=0.25, iou_thres=0.45, imgW=608, imgH=608, ratio=1.0, dw=0.0, dh=0.0):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    img_scale = [imgW / INPUT_WIDTH, imgH / INPUT_HEIGHT, imgW / INPUT_WIDTH, imgH / INPUT_HEIGHT]

    t = time.time()
    detected_objects = []
    for xi, x in enumerate(prediction):  # image index, image inference
        batch_objects = []
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        #box = x[:, :4]

        # Detections matrix nx6 (xyxy, conf, cls)

        #test1 = np.max(x[:, 5:], 1, keepdims=True)
        conf = np.max(x[:, 5:], 1, keepdims=True)
        j = np.argmax(x[:, 5:], 1)
        labels = set(j.astype(int))
        j = np.expand_dims(j, axis=1)
        j = j.astype("float32")
        #x = np.hstack(box, conf, j)
        x = np.concatenate((box, conf, j), axis=1)

        for label in labels:
            selected_bboxes = x[np.where((x[:, 5] == label) & (x[:, 4] >= conf_thres))]
            selected_bboxes_keep = selected_bboxes[nms(selected_bboxes[:, :4], selected_bboxes[:, 4], iou_thres)]
            for idx in range(selected_bboxes_keep.shape[0]):
                box_xy = (selected_bboxes_keep[idx, :2] - [dw, dh]) / ratio
                box_xy1 = (selected_bboxes_keep[idx, 2:4] - [dw, dh]) / ratio
                score = selected_bboxes_keep[idx, 4]

                batch_objects.append([label, score, box_xy[0], box_xy[1], box_xy1[0], box_xy1[1]])
        detected_objects.append(batch_objects)
    return detected_objects

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

_LINE_THICKNESS_SCALING = 500.0

def render_box(img, box, color=(200, 200, 200)):
    """
    Render a box. Calculates scaling and thickness automatically.
    :param img: image to render into
    :param box: (x1, y1, x2, y2) - box coordinates
    :param color: (b, g, r) - box color
    :return: updated image
    """
    x1, y1, x2, y2 = box
    thickness = int(
        round(
            (img.shape[0] * img.shape[1])
            / (_LINE_THICKNESS_SCALING * _LINE_THICKNESS_SCALING)
        )
    )
    thickness = max(1, thickness)
    img = cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness=thickness
    )
    return img

_TEXT_THICKNESS_SCALING = 700.0
_TEXT_SCALING = 520.0


def get_text_size(img, text, normalised_scaling=1.0):
    """
    Get calculated text size (as box width and height)
    :param img: image reference, used to determine appropriate text scaling
    :param text: text to display
    :param normalised_scaling: additional normalised scaling. Default 1.0.
    :return: (width, height) - width and height of text box
    """
    thickness = int(
        round(
            (img.shape[0] * img.shape[1])
            / (_TEXT_THICKNESS_SCALING * _TEXT_THICKNESS_SCALING)
        )
        * normalised_scaling
    )
    thickness = max(1, thickness)
    scaling = img.shape[0] / _TEXT_SCALING * normalised_scaling
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scaling, thickness)[0]

def render_text(img, text, pos, color=(200, 200, 200), normalised_scaling=1.0):
    """
    Render a text into the image. Calculates scaling and thickness automatically.
    :param img: image to render into
    :param text: text to display
    :param pos: (x, y) - upper left coordinates of render position
    :param color: (b, g, r) - text color
    :param normalised_scaling: additional normalised scaling. Default 1.0.
    :return: updated image
    """
    x, y = pos
    thickness = int(
        round(
            (img.shape[0] * img.shape[1])
            / (_TEXT_THICKNESS_SCALING * _TEXT_THICKNESS_SCALING)
        )
        * normalised_scaling
    )
    thickness = max(1, thickness)
    scaling = img.shape[0] / _TEXT_SCALING * normalised_scaling
    size = get_text_size(img, text, normalised_scaling)
    cv2.putText(
        img,
        text,
        (int(x), int(y + size[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        scaling,
        color,
        thickness=thickness,
    )
    return img

# 调用入口为集成模型
model_name = "yolov5s-onnx"

if __name__ == "__main__":
    with httpclient.InferenceServerClient("192.168.2.230:8000") as client:
        #input0_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
        input0_data = cv2.imread("E:/haige/yolov5/data/images/20210811144856.jpg")
        # Padded resize
        img, ratio, (dw, dh) = letterbox(input0_data, (640, 640))
        #cv2.imshow("img", img)
        #cv2.waitKey(0)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32')
        img /= 255.0
        img = np.ascontiguousarray(img)
        inputs = [
            httpclient.InferInput("images", img.shape, np_to_triton_dtype(img.dtype))
        ]

        inputs[0].set_data_from_numpy(img)

        outputs = [httpclient.InferRequestedOutput("output")]

        response = client.infer(model_name,
                                inputs,
                                request_id=str(1),
                                outputs=outputs)
        #result = response.get_response()
        result = response.as_numpy("output")
        pred = postprecess(result, 0.6, 0.5, input0_data.shape[1], input0_data.shape[0], ratio, dw, dh)
        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            if len(det):
                for box in det:
                    render_box(input0_data, box[2:], color=tuple(RAND_COLORS[box[0] % 64].tolist()))
                    render_text(input0_data, f"{COCOLabels(box[0]).name}: {box[1]:.2f}", (box[2], box[3]), color=(0, 0, 255), normalised_scaling=0.5)
            cv2.imshow("draw", input0_data)
            cv2.waitKey(0)
