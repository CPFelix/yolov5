"""
https://blog.csdn.net/qq_39686950/article/details/119153685?spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-12.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-12.no_search_link&utm_relevant_index=18
"""
import os
import json
import cv2
import random
import time
from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":
    coco_format_save_path = 'F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang_2+DSMhand_smoke3\\train\\json\\'  # 要生成的标准coco格式标签所在文件夹
    yolo_format_classes_path = 'E:\\pycharm-projects\\dataset\\DSMhand_smoke3\\yolov5\\class_name.txt'  # 类别文件，一行一个类
    yolo_format_annotation_path = 'E:\\pycharm-projects\\dataset\\DSMhand_smoke3\\yolov5\\labels\\train\\'  # yolo格式标签所在文件夹
    img_pathDir = 'E:\\pycharm-projects\\dataset\\DSMhand_smoke3\\yolov5\\images\\train\\'  # 图片所在文件夹

    with open(yolo_format_classes_path, 'r') as fr:  # 打开并读取类别文件
        lines1 = fr.readlines()

    # 接下来的代码主要添加'images'和'annotations'的key值
    imageFileList = os.listdir(img_pathDir)  # 遍历该文件夹下的所有文件，并将所有文件名添加到列表中
    for i, imageFile in enumerate(tqdm(imageFileList)):
        imagePath = os.path.join(img_pathDir, imageFile)  # 获取图片的绝对路径
        image = Image.open(imagePath)  # 读取图片，然后获取图片的宽和高
        W, H = image.size

        new_dict = {"version": "4.6.0", "flags": {}, "shapes": [],
                    "imageData": None}  # label format
        new_dict["imagePath"] = imageFile
        temp_img = cv2.imread(imagePath)
        height, width = temp_img.shape[:2]
        new_dict["imageHeight"] = height
        new_dict["imageWidth"] = width

        txtFile = imageFile.split(".")[0] + '.txt'  # 获取该图片获取的txt文件
        with open(os.path.join(yolo_format_annotation_path, txtFile), 'r') as fr:
            lines = fr.readlines()  # 读取txt文件的每一行数据，lines2是一个列表，包含了一个图片的所有标注信息
        for j, line in enumerate(lines):
            bbox_dict = {}  # 将每一个bounding box信息存储在该字典中
            # line = line.strip().split()
            # print(line.strip().split(' '))

            class_id, x, y, w, h = line.strip().split(' ')  # 获取每一个标注框的详细信息
            class_id, x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)  # 将字符串类型转为可计算的int和float类型

            xmin = (x - w / 2) * W  # 坐标转换
            ymin = (y - h / 2) * H
            xmax = (x + w / 2) * W
            ymax = (y + h / 2) * H
            w = w * W
            h = h * H
            object = {}
            if (class_id == 0):
                object["label"] = "hand"
                object["group_id"] = None
                object["shape_type"] = "rectangle"
                object["flags"] = {}
                object["points"] = []
                x1y1 = [xmin, ymin]
                x2y2 = [xmax, ymax]
                object["points"].append(x1y1)
                object["points"].append(x2y2)
            elif (class_id == 1):
                object["label"] = "cigarette"
                object["group_id"] = None
                object["shape_type"] = "rectangle"
                object["flags"] = {}
                object["points"] = []
                x1y1 = [xmin, ymin]
                x2y2 = [xmax, ymax]
                object["points"].append(x1y1)
                object["points"].append(x2y2)
        new_dict["shapes"].append(object)

        json_path = os.path.join(coco_format_save_path, imageFile.split(".")[0])
        with open(json_path + '.json', 'a') as f:
            json.dump(new_dict, f, indent=4)

