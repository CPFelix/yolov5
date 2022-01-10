import argparse
import numpy as np
import sys
from builtins import range

import tritonclient.http as httpclient
import tritonclient.utils.shared_memory as shm
from tritonclient import utils

from processing import preprocess, postprocess
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from labels import COCOLabels
from tqdm import tqdm
import os
import cv2
import time

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')

    FLAGS = parser.parse_args()

    try:
        triton_client = httpclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    # To make sure no shared memory regions are registered with the
    # server.
    triton_client.unregister_system_shared_memory()
    triton_client.unregister_cuda_shared_memory()

    # We use a simple model that takes 2 input tensors of 16 integers
    # each and returns 2 output tensors of 16 integers each. One
    # output tensor is the element-wise sum of the inputs and one
    # output is the element-wise difference.
    model_name = "yolov5s-trt"
    model_version = ""

    # imgDir = "E:/haige/yolov5/data/CocoTest100"
    imgDir = "/home/chenpengfei/yolov5/data/CocoTest100"
    totalTime = 0.0
    for filename in tqdm(os.listdir(imgDir)):
        # print(filename)
        imgPath = os.path.join(imgDir, filename)
        # input0_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
        input0_data = cv2.imread(imgPath)
        # Padded resize
        img, ratio, (dw, dh) = preprocess(input0_data)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        # Create the data for the two input tensors. Initialize the first
        # to unique integers and the second to all ones.
        input0_data = img
        #input1_data = np.ones(shape=16, dtype=np.int32)
        output0_data = np.ones(shape=[1,6001,1,1], dtype=np.float32)

        input_byte_size = input0_data.size * input0_data.itemsize
        output_byte_size = output0_data.size * output0_data.itemsize

        # Create Output0 and Output1 in Shared Memory and store shared memory handles
        shm_op0_handle = shm.create_shared_memory_region("output0_data", byte_size = output_byte_size)
        # shm_op1_handle = shm.create_shared_memory_region("output1_data",
        #                                                  "/output1_simple",
        #                                                  output_byte_size)

        # Register Output0 and Output1 shared memory with Triton Server
        triton_client.register_system_shared_memory("output0_data", byte_size = output_byte_size)
        # triton_client.register_system_shared_memory("output1_data",
        #                                             "/output1_simple",
        #                                             output_byte_size)

        # Create Input0 and Input1 in Shared Memory and store shared memory handles
        shm_ip0_handle = shm.create_shared_memory_region("input0_data", byte_size = input_byte_size)
        # shm_ip1_handle = shm.create_shared_memory_region("input1_data",
        #                                                  "/input1_simple",
        #                                                  input_byte_size)

        # Put input data values into shared memory
        shm.set_shared_memory_region(shm_ip0_handle, [input0_data])
        #shm.set_shared_memory_region(shm_ip1_handle, [input1_data])

        # Register Input0 and Input1 shared memory with Triton Server
        triton_client.register_system_shared_memory("input0_data", byte_size = input_byte_size)
        # triton_client.register_system_shared_memory("input1_data", "/input1_simple",
        #                                             input_byte_size)

        t0 = time.time()
        # Set the parameters to use data from shared memory
        inputs = []
        inputs.append(httpclient.InferInput('data', input0_data.shape, "FP32"))
        inputs[-1].set_shared_memory("input0_data", byte_size = input_byte_size)

        # inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))
        # inputs[-1].set_shared_memory("input1_data", input_byte_size)

        outputs = []
        outputs.append(httpclient.InferRequestedOutput('prob', binary_data=True))
        outputs[-1].set_shared_memory("output0_data", byte_size = output_byte_size)

        # outputs.append(httpclient.InferRequestedOutput('OUTPUT1', binary_data=True))
        # outputs[-1].set_shared_memory("output1_data", output_byte_size)

        results = triton_client.infer(model_name=model_name,
                                      inputs=inputs,
                                      outputs=outputs)

        # Read results from the shared memory.
        output0 = results.get_output("prob")
        if output0 is not None:
            output0_data = shm.get_contents_as_numpy(
                shm_op0_handle, utils.triton_to_np_dtype(output0['datatype']),
                output0['shape'])
        else:
            print("prob is missing in the response.")
            sys.exit(1)

        t1 = time.time()

        totalTime += (t1 - t0) * 1000.0

        detected_objects = postprocess(output0, ratio, dw, dh)
        # print(f"Raw boxes: {int(result[0, 0, 0, 0])}")
        # print(f"Detected objects: {len(detected_objects)}")

        input_image = input0_data.copy()
        for box in detected_objects:
            # print(f"{COCOLabels(box[0]).name}: {box[1]}")

            input_image = render_box(input_image, box[2:], color=tuple(RAND_COLORS[box[0] % 64].tolist()))
            size = get_text_size(input_image, f"{COCOLabels(box[0]).name}: {box[1]:.2f}",
                                 normalised_scaling=0.6)
            input_image = render_filled_box(input_image, (box[2] - 3, box[3] - 3, box[2] + size[0], box[3] + size[1]),
                                            color=(220, 220, 220))
            input_image = render_text(input_image, f"{COCOLabels(box[0]).name}: {box[1]:.2f}",
                                      (box[2], box[3]), color=(30, 30, 30), normalised_scaling=0.5)

        cv2.imwrite("./drawImg/" + filename, input_image)
        print(f"Saved result to {FLAGS.out}")



        # output1 = results.get_output("OUTPUT1")
        # if output1 is not None:
        #     output1_data = shm.get_contents_as_numpy(
        #         shm_op1_handle, utils.triton_to_np_dtype(output1['datatype']),
        #         output1['shape'])
        # else:
        #     print("OUTPUT1 is missing in the response.")
        #     sys.exit(1)

        # for i in range(16):
        #     print(
        #         str(input0_data[i]) + " + " + str(input1_data[i]) + " = " +
        #         str(output0_data[0][i]))
        #     print(
        #         str(input0_data[i]) + " - " + str(input1_data[i]) + " = " +
        #         str(output1_data[0][i]))
        #     if (input0_data[i] + input1_data[i]) != output0_data[0][i]:
        #         print("shm infer error: incorrect sum")
        #         sys.exit(1)
        #     if (input0_data[i] - input1_data[i]) != output1_data[0][i]:
        #         print("shm infer error: incorrect difference")
        #         sys.exit(1)

        print(triton_client.get_system_shared_memory_status())
        triton_client.unregister_system_shared_memory()
        assert len(shm.mapped_shared_memory_regions()) == 4
        shm.destroy_shared_memory_region(shm_ip0_handle)
        #shm.destroy_shared_memory_region(shm_ip1_handle)
        shm.destroy_shared_memory_region(shm_op0_handle)
        #shm.destroy_shared_memory_region(shm_op1_handle)
        assert len(shm.mapped_shared_memory_regions()) == 0

        print('PASS: system shared memory')

    print(f'TotalTime: ({totalTime:.3f}ms)')