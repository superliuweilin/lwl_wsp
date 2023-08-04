import cv2
import numpy as np
#
# # 读取原始单通道PNG数据
# image = cv2.imread('/home/lyu3/lwl_wp/open-cd/datasets/CD/train_with_seg/train/label/1.png', cv2.IMREAD_GRAYSCALE)
#
# # 将原始数据映射到目标像素值
# converted_image = np.where(image == 0, 0, 255).astype(np.uint8)
#
# # 创建3通道的图像
# output_image = np.stack([converted_image] * 3, axis=-1)
#
# # 保存转换后的图像
# cv2.imwrite('/home/lyu3/lwl_wp/open-cd/datasets/CD/train_with_seg/train/label_converted/1.png', output_image)

import os
import cv2
import numpy as np

# 定义原始数据文件夹和转换后数据保存文件夹的路径
original_folder = '/home/lyu/datasets/CD/train_with_seg/train/label'
converted_folder = '/home/lyu/datasets/CD/train_with_seg/train/label_converted'

# 确保转换后数据保存文件夹存在
os.makedirs(converted_folder, exist_ok=True)

# 遍历原始数据文件夹中的所有图片
for filename in os.listdir(original_folder):
    if filename.endswith('.png'):
        # 构建原始数据文件路径
        original_path = os.path.join(original_folder, filename)

        # 读取原始单通道PNG数据
        image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

        # 将原始数据映射到目标像素值
        converted_image = np.where(image == 0, 0, 255).astype(np.uint8)

        # 创建3通道的图像
        output_image = np.stack([converted_image] * 3, axis=-1)

        # 构建转换后数据保存路径
        converted_path = os.path.join(converted_folder, filename)

        # 保存转换后的图像
        cv2.imwrite(converted_path, output_image)

