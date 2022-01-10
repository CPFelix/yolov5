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

import os
from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np
import time
import cv2

from processing import preprocess, postprocess
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS

from labels import COCOLabels
from tqdm import tqdm

# 调用入口为集成模型
model_name = "ensemble_yolov5s-trt_post"

if __name__ == "__main__":
    with httpclient.InferenceServerClient("192.168.2.230:8000") as client:
        #imgDir = "E:/haige/yolov5/data/CocoTest100"
        imgDir = "/home/chenpengfei/yolov5/data/CocoTest100"
        totalTime = 0.0
        for filename in tqdm(os.listdir(imgDir)):
            #print(filename)
            imgPath = os.path.join(imgDir, filename)
            #input0_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
            input0_data = cv2.imread(imgPath)
            # Padded resize
            img, ratio, (dw, dh) = preprocess(input0_data)
            #cv2.imshow("img", img)
            #cv2.waitKey(0)

            t0 = time.time()

            padding = np.array([[dw, dh, ratio[0]]]).astype("float32")
            inputs = [
                httpclient.InferInput("IMAGE", img.shape, np_to_triton_dtype(img.dtype)),
                httpclient.InferInput("PAD", padding.shape, np_to_triton_dtype(padding.dtype))
            ]

            inputs[0].set_data_from_numpy(img)
            inputs[1].set_data_from_numpy(padding)

            outputs = [httpclient.InferRequestedOutput("BBOXES")]

            response = client.infer(model_name,
                                    inputs,
                                    request_id=str(1),
                                    outputs=outputs)
            #result = response.get_response()
            result = response.as_numpy("BBOXES")

            t1 = time.time()

            totalTime += (t1 - t0) * 1000.0

            input_image = input0_data.copy()
            if (result.shape[0] != 0):
                for box in result[0]:
                    #print(f"{COCOLabels(int(box[0])).name}: {box[1]}")

                    input_image = render_box(input_image, box[2:], color=tuple(RAND_COLORS[int(box[0]) % 64].tolist()))
                    size = get_text_size(input_image, f"{COCOLabels(int(box[0])).name}: {box[1]:.2f}",
                                         normalised_scaling=0.6)
                    input_image = render_filled_box(input_image, (box[2] - 3, box[3] - 3, box[2] + size[0], box[3] + size[1]),
                                                    color=(220, 220, 220))
                    input_image = render_text(input_image, f"{COCOLabels(int(box[0])).name}: {box[1]:.2f}",
                                              (box[2], box[3]), color=(30, 30, 30), normalised_scaling=0.5)

                # cv2.imwrite(FLAGS.out, input_image)
                # print(f"Saved result to {FLAGS.out}")

                # cv2.imshow('image', input_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            else:
                pass
                #print("No objects detected!")
                # cv2.imshow('image', input_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

        print(f'TotalTime: ({totalTime:.3f}ms)')