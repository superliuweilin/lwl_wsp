import os

from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import pycocotools.mask as mutils
from opencd.models import *
from mmengine.runner import Runner
from mmengine.config import Config


# # norm_cfg = dict(type='SyncBN', requires_grad=True)
# # cfg = dict(
# #     type='DIEncoderDecoder',
# #     data_preprocessor=data_preprocessor,
# #     pretrained=None,
# #     backbone=dict(
# #         type='IA_ResNeSt',
# #         depth=101,
# #         num_stages=4,
# #         out_indices=(0, 1, 2, 3),
# #         dilations=(1, 1, 1, 1),
# #         strides=(1, 2, 2, 2),
# #         norm_cfg=norm_cfg,
# #         norm_eval=False,
# #         style='pytorch',
# #         contract_dilation=True,
# #         stem_channels=128,
# #         radix=2,
# #         reduction_factor=4,
# #         avg_down_stride=True,
# #         interaction_cfg=(None,
# #             dict(type='SpatialExchange', p=1/2),
# #             dict(type='ChannelExchange', p=1/2),
# #             dict(type='ChannelExchange', p=1/2))),
# #     decode_head=dict(
# #         type='Changer',
# #         in_channels=[256, 512, 1024, 2048],
# #         in_index=[0, 1, 2, 3],
# #         channels=256,
# #         dropout_ratio=0.1,
# #         num_classes=2,
# #         norm_cfg=norm_cfg,
# #         align_corners=False,
# #         sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000),
# #         loss_decode=dict(
# #             type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
# #     # model training and testing settings
# #     train_cfg=dict(),
# #     test_cfg=dict(mode='whole'))

data_dir = "/home/lyu/lwl_wsp/open-cd/datasets/CD/testA"
train_images_A = sorted(glob.glob(os.path.join(data_dir, "A/*")))
train_images_B = sorted(glob.glob(os.path.join(data_dir, "B/*")))
df = pd.DataFrame({"image_file_A": train_images_A, "image_file_B": train_images_B})
df["uid"] = df.image_file_A.apply(lambda x: int(os.path.basename(x).split(".")[0]))
config = "/home/lyu/lwl_wsp/open-cd/configs/tinycd/tinycd_256x256_40k_levircd.py"
model_info = "/home/lyu/lwl_wsp/open-cd/tinycd_GF_CD_workdir/best_mIoU_iter_40000.pth"
cfg = Config.fromfile(config)
# Runner.build_model()
runner = Runner.from_cfg(cfg)
models = runner.model
state_dict = torch.load(model_info)
# for key in state_dict:
#     print(key)
models.load_state_dict(state_dict['state_dict'])
models = models.eval()
# input = torch.cuda.FloatTensor(np.random.rand(1, 6, 512, 512))
# out = models(input)
# print('out.shape', out.shape)



# def get_model(cfg):
#     cfg = cfg.copy()
#     model = eval(cfg.pop("type"))(**cfg)
#     return model
#
#
# def get_models():
#     # model_infos = [
#     #     dict(
#     #         ckpt=f"./logs/{name}/f{fold}/last.ckpt",
#     #     ) for name in names for fold in folds
#     # ]
#     model_info = "/home/lyu3/lwl_wp/open-cd/changer_s101_GF_CD_workdir/best_mIoU_iter_4000.pth"
#     # models = []
#     # for model_info in model_infos:
#     #     if not os.path.exists(model_info["pth"]):
#     #         model_info['pth'] = sorted(glob.glob(model_info['pth']))[-1]
#     #     stt = torch.load(model_info["pth"], map_location="cpu")
#     #     cfg = OmegaConf.create(eval(str(stt["hyper_parameters"]))).model
#     #     stt = {k[6:]: v for k, v in stt["state_dict"].items()}
#     #
#     #     model = get_model(cfg)
#     #     model.load_state_dict(stt, strict=True)
#     #     model.eval()
#     #     model.cuda()
#     stt = torch.load(model_info, map_location="cpu")
#     # for key in stt.keys():
#     #     print(key)
#     # print(stt["meta"])
#     # cfg = OmegaConf.create(eval(str(stt["meta"]))).model
#     # stt = {k[6:]: v for k, v in stt["state_dict"].items()}
#
#     # model = get_model(cfg)
#     model.load_state_dict(stt["state_dict"], strict=True)
#     model.eval()
#     model.cuda()
#     # models.append(model)
#     return model
#
#
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def load(row):
    imgA = cv2.imread(row.image_file_A)
    imgB = cv2.imread(row.image_file_B)
    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
    imgA = (imgA / 255. - mean) / std
    imgB = (imgB / 255. - mean) / std
    img = np.concatenate([imgA, imgB], -1).astype(np.float32)
    return img, None


def predict(row, models, img):
    img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
        preds = []
        pred = models(img).sigmoid()
        pred = pred.squeeze().detach().cpu().numpy()
        preds.append(pred)
        pred = sum(preds) / len(preds)
    return pred


def get_dt(row, pred, img_id, dts):

    mask = pred.round().astype(np.uint8)
    # print(mask)
    # print(mask.shape)

    nc, label = cv2.connectedComponents(mask, connectivity=8)
    # print(nc, label)
    # print(nc, label)
    print(nc)
    print(label)
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


# names = [
#     "base"
# ]
# folds = [0]

os.system("mkdir -p results")
sub = df
# models = get_models(names, folds)
dts = []
# print(len(sub))
for idx in tqdm(range(len(sub))):
    row = sub.loc[idx]
    img, mask = load(row)
    # print(row)

    pred = predict(row, models, img)
    # print(pred.shape)
    get_dt(row, pred, row.uid, dts)
with open("./results/test.segm.json", "w") as f:
    json.dump(dts, f)
os.system("zip -9 -r results.zip results")