import json
import os
import cv2
import numpy as np
from tqdm import tqdm

# 将labelme标注文件转化为yolov5所需数据格式
if __name__ == "__main__":
    jsonPath = "F:\\阜康测试视频\\frame-16\\labelme\\test\\json"
    txtPath = "F:\\阜康测试视频\\frame-16\\labelme\\test\\labels"
    imgPath = "F:\\阜康测试视频\\frame-16\\labelme\\test\\images"
    for jsonname in tqdm(os.listdir(jsonPath)):
        jsonfile = os.path.join(jsonPath, jsonname)
        txtfile = os.path.join(txtPath, jsonname.split(".")[0] + ".txt")
        imagefile = os.path.join(imgPath, jsonname.split(".")[0] + ".jpg")
        image = cv2.imdecode(np.fromfile(imagefile, dtype=np.uint8), -1)
        height, width = image.shape[:2]
        with open(jsonfile, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)
            # print(json_data)
            with open(txtfile, "w", encoding='utf8') as ft:
                for object in json_data["shapes"]:
                    if (object["label"] == "hand"):
                        label = 0
                    elif (object["label"] == "cigarette"):
                        label = 1
                    x1 = object["points"][0][0]
                    y1 = object["points"][0][1]
                    x2 = object["points"][1][0]
                    y2 = object["points"][1][1]
                    object_x_center = (int(x1) + int(x2)) / 2.0 / float(width)
                    object_y_center = (int(y1) + int(y2)) / 2.0 / float(height)
                    object_width = (int(x2) - int(x1)) / float(width)
                    object_height = (int(y2) - int(y1)) / float(height)
                    ft.write(str(label) + " " + str(object_x_center) + " " + str(object_y_center) + " " + str(object_width) + " " + str(object_height) + "\n")

