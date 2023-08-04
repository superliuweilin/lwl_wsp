import os
import cv2
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import pycocotools.mask as mutils
from argparse import ArgumentParser
import zipfile


def get_dt(row, pred, img_id, dts):
    mask = pred.round().astype(np.uint8)
    nc, label = cv2.connectedComponents(mask, connectivity=8)
    for c in range(nc):
        if np.all(mask[label == c] == 0):
            continue
        else:
            ann = np.asfortranarray((label == c).astype(np.uint8))
            rle = mutils.encode(ann)
            bbox = [int(_) for _ in mutils.toBbox(rle)]
            area = int(mutils.area(rle))
            score = float(pred[label == c].mean())
            dts.append({
                "segmentation": {
                    "size": [int(_) for _ in rle["size"]],
                    "counts": rle["counts"].decode()},
                "bbox": [int(_) for _ in bbox], "area": int(area), "iscrowd": 0, "category_id": 1,
                "image_id": int(img_id), "id": len(dts),
                "score": float(score)
            })


def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')

        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join('results', filename))

    zip.close()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--results_path', type=str, required=True, help='Add the resulting image to be processed')
    parser.add_argument('--out_zip_path', type=str, required=True, help='Add the output image path')
    args = parser.parse_args()

    # args.results_path = r'C:\1a_runing\change\sub\results'
    # args.out_zip_path = r'C:\1a_runing\change\sub\out'

    # 对已有结果进行处理
    pred_path = args.results_path
    save_json = os.path.join(pred_path, 'test.segm.json')

    if os.path.exists(save_json):
        os.remove(save_json)

    if not os.path.isdir(args.out_zip_path):
        os.makedirs(args.out_zip_path)

    df_results = sorted(glob.glob(os.path.join(pred_path, "*")))

    df = pd.DataFrame({"df_results": df_results})
    df["uid"] = df.df_results.apply(lambda x: os.path.basename(x).split(".")[0])

    sub = df

    dts = []

    for idx in tqdm(range(len(sub))):
        row = sub.loc[idx]
        uid = row.uid
        pred_name = os.path.join(pred_path, str(uid) + '.png')

        pred = cv2.imread(pred_name, cv2.IMREAD_UNCHANGED)
        get_dt(row, pred, row.uid, dts)

    with open(save_json, "w") as f:
        json.dump(dts, f)

    input_path = args.results_path
    output_path = os.path.join(args.out_zip_path, 'results.zip')

    if os.path.exists(output_path):
        os.remove(output_path)

    zipDir(input_path, output_path)