import cv2 as cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from builder import parser_model

samples = {
    24: [
        '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/DukeMTMC-reID/bounding_box_train/0024_c4_f0032417.jpg',
        '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/DukeMTMC-reID/bounding_box_train/0024_c1_f0052855.jpg',
        '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/DukeMTMC-reID/bounding_box_train/0040_c2_f0059190.jpg',
        '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/DukeMTMC-reID/bounding_box_train/0096_c8_f0029666.jpg',
        '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/DukeMTMC-reID/bounding_box_train/0146_c4_f0050488.jpg',
    ]
    # 180: [
    #     '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/preprocessed-shuffle-22/task-0-0/train/180/0200_c5_f0091943.jpg',
    #     '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/preprocessed-shuffle-22/task-2-0/train/180/0200_c1_f0089462.jpg',
    #     '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/preprocessed-shuffle-22/task-3-0/train/180/0200_c2_f0089200.jpg',
    #     '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/preprocessed-shuffle-22/task-4-0/train/180/0200_c6_f0067466.jpg',
    # ],
    # 128: [
    #     '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/preprocessed-shuffle-22/task-0-0/train/128/2953_c5_f0066209.jpg',
    #     '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/preprocessed-shuffle-22/task-1-0/train/128/2953_c4_f0037049.jpg',
    #     '/home/zhanglei/projects/Awesome-ReID-for-FCL/datasets/preprocessed-shuffle-22/task-4-0/train/128/2953_c6_f0042312.jpg',
    # ]
}

models = {
    '../configs/basis_exp/experiment_fedstil.yaml': '../ckpts/2022-3-1/fedstil_k_9/client-0/fedstil_model.ckpt',
    '../configs/basis_exp/experiment_mas.yaml': '../ckpts_bak/2022-3-1/mas/client-0/mas_model.ckpt',
    '../configs/basis_exp/experiment_fedprox.yaml': '../ckpts_bak/2022-3-1/fedprox/client-0/fedprox_model.ckpt',
}

save_dir = '../logs/cam'

if __name__ == '__main__':

    for method_config, model_ckpt in models.items():
        with open(method_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        method_name = config['exp_name']
        method_type = config['exp_method']
        model = parser_model(method_type, config['model_opts'])
        model.update_model(torch.load(model_ckpt))

        target_layers = [model.net.base.layer4[-1]]
        cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=False)

        for class_id, sample_paths in samples.items():
            for img_id, img_path in enumerate(sample_paths):
                rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
                rgb_img = np.float32(rgb_img) / 255

                input_tensor = transforms.ToTensor()(rgb_img).unsqueeze(0)
                target_category = [ClassifierOutputTarget(class_id)]
                grayscale_cam = cam(input_tensor, target_category)[0, :]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)
                cv2.imwrite(f'{save_dir}/{method_name}_{class_id}_{img_id}.jpg', visualization)
