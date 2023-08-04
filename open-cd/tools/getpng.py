import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mmengine.runner import Runner
from mmengine.config import Config
from torchvision.datasets import ImageFolder


def GetModelLoadWeight(config, model_info):

    cfg = Config.fromfile(config)
    runner = Runner.from_cfg(cfg)
    models = runner.model
    state_dict = torch.load(model_info)
    # for key in state_dict:
    #     print(key)
    models.load_state_dict(state_dict['state_dict'])
    models = models.eval()
    return  models

# 测试图片并保存生成结果
def test_and_save_images(model, test_loader):
    with torch.no_grad():
        for data in test_loader:
            images, _ = data
            images = images.view(images.size(0), -1)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), 1, 28, 28)  # 由于MNIST图片是单通道的，需要重新调整形状
            for i in range(outputs.size(0)):
                plt.imshow(outputs[i].squeeze().numpy(), cmap='gray')
                plt.savefig(f'test_result_{i}.png')

if __name__ == '__main__':
    # 设置参数
    data_dir = "/home/lyu/lwl_wsp/open-cd/datasets/CD/testA"

    config = "/home/lyu/lwl_wsp/open-cd/configs/tinycd/tinycd_256x256_40k_levircd.py"
    model_info = "/home/lyu/lwl_wsp/open-cd/tinycd_GF_CD_workdir/best_mIoU_iter_40000.pth"
    model = GetModelLoadWeight(config, model_info)
    # 加载MNIST测试集
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


    # 测试并保存生成结果
    test_and_save_images(model, test_loader)
