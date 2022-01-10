# 获取烟遮挡数据的手部区域和对应烟坐标用于训练烟头检测
import os
import csv
from shutil import copyfile
import cv2
import numpy as np
import random

if __name__ == "__main__":
    # imgPath = "E:\\pycharm-projects\\dataset\\检测数据集\\keruisite\\sigurate"
    # csvfile = "E:\\pycharm-projects\\dataset\\检测数据集\\keruisite\\sigurate_smoke.csv"
    # ImgsavePath = "E:\\pycharm-projects\\dataset\\DSMhand&smoke2\\temp\\keruisite\\sigurate"
    # TxtsavePath = "E:\\pycharm-projects\\dataset\\DSMhand&smoke2\\temp\\keruisite\\sigurate"
    # with open(csvfile, 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         print(row)
    #         imgfile = os.path.join(imgPath, row[0])
    #         img = cv2.imdecode(np.fromfile(imgfile, dtype=np.uint8), 1)
    #         width = img.shape[1]
    #         height = img.shape[0]
    #         target_file = os.path.join(ImgsavePath, row[0])
    #         if not os.path.exists(ImgsavePath):
    #             os.makedirs(ImgsavePath)
    #         copyfile(imgfile, target_file)
    #         txtfile = os.path.join(TxtsavePath, row[0].split(".")[0] + ".txt")
    #         with open(txtfile, "a") as f1:
    #             # hand 0  smoke 1
    #             label = 1
    #             object_x_center = (int(row[1]) + int(row[3])) / 2.0 / float(width)
    #             object_y_center = (int(row[2]) + int(row[4])) / 2.0 / float(height)
    #             object_width = (int(row[3]) - int(row[1])) / float(width)
    #             object_height = (int(row[4]) - int(row[2])) / float(height)
    #             f1.write(str(label) + " " + str(object_x_center) + " " + str(object_y_center) + " " + str(object_width) + " " + str(object_height) + "\n")
    # dict_id = {}
    # for imgname in os.listdir("E:\\pycharm-projects\\dataset\\DSMhand_smoke3\\images\\train\\mp"):
    #     # txtname = imgname.split(".")[0] + ".txt"
    #     # source_file = os.path.join("E:\\pycharm-projects\\dataset\\DSMhand&smoke1\\labels\\train", txtname)
    #     # target_file = os.path.join("E:\\pycharm-projects\\dataset\\DSMhand&smoke1\\labels\\val", txtname)
    #     # copyfile(source_file, target_file)
    #     # os.remove(source_file)
    #     ID = int(imgname.split("_")[2][2:])
    #     if (ID in dict_id.keys()):
    #         dict_id[ID].append(imgname)
    #     else:
    #         dict_id[ID] = []
    #         dict_id[ID].append(imgname)
    #
    # # 随机选择40个ID做测试集
    # val_id = random.sample(dict_id.keys(), 40)
    # print(val_id)
    # for id in val_id:
    #     for imgname in dict_id[id]:
    #         source_file = "E:\\pycharm-projects\\dataset\\DSMhand_smoke3\\images\\train\\mp\\" + imgname
    #         target_file = "E:\\pycharm-projects\\dataset\\DSMhand_smoke3\\images\\val\\" + imgname
    #         copyfile(source_file, target_file)
    #         os.remove(source_file)
    # for imgname in os.listdir("E:\\pycharm-projects\\dataset\\DSMhand_smoke3\\images\\val"):
    #     txtfile = "E:\\pycharm-projects\\dataset\\DSMhand_smoke3\\labels\\train\\" + imgname.split(".")[0] + ".txt"
    #     if os.path.exists(txtfile):
    #         source_file = txtfile
    #         target_file = "E:\\pycharm-projects\\dataset\\DSMhand_smoke3\\labels\\val\\" + imgname.split(".")[0] + ".txt"
    #         copyfile(source_file, target_file)
    #         os.remove(source_file)

    # 删除没有json的图片
    for imgname in os.listdir("F:\\阜康测试视频\\frame-16\\labelme\\test\\img"):
        txtfile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\json\\" + imgname.split(".")[0] + ".json"
        if not os.path.exists(txtfile):
            imagefile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\img\\" + imgname
            print(imagefile)
            os.remove(imagefile)

    # for jsonname in os.listdir("F:\\阜康测试视频\\frame-16\\labelme\\test\\1\\json"):
    #     imagefile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\1\\img\\" + jsonname.split(".")[0] + ".jpg"
    #     if not os.path.exists(imagefile):
    #         jsonfile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\1\\json\\" + jsonname
    #         print(jsonfile)
    #         os.remove(jsonfile)



