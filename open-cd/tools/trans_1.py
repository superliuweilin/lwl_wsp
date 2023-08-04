import os
import cv2
import numpy as np

# 定义原始数据文件夹和还原后数据保存文件夹的路径
original_folder = '/home/lyu/lwl_wsp/open-cd/work_dirs/lwl_work_dir/changer_s101_512x512_80k/vis_data/vis_image'
restored_folder = '/home/lyu/lwl_wsp/open-cd/work_dirs/lwl_work_dir/changer_s101_512x512_80k/vis_data/vis_image_trans'

# 确保还原后数据保存文件夹存在
os.makedirs(restored_folder, exist_ok=True)

# 遍历数据文件夹中的所有图片
for filename in os.listdir(original_folder):
    if filename.endswith('.png'):
        # 构建转换后数据文件路径
        original_path = os.path.join(original_folder, filename)

        # 读取单通道PNG数据
        image = cv2.imread(original_path)
        

        restored_image = image[:, :, 0]
        

        # 将像素值映射到（0，1）范围
        restored_image = restored_image / 255.0

        # 构建还原后数据保存路径
        restored_path = os.path.join(restored_folder, filename)

        # 保存还原后的单通道图像
        cv2.imwrite(restored_path, restored_image.astype(np.uint8))

print('over!')
