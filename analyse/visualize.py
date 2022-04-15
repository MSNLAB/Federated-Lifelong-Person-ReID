import cv2 as cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from builder import parser_model

samples = {
    # 24: [
    #     './DukeMTMC-reID/bounding_box_train/0024_c4_f0032417.jpg',
    #     './DukeMTMC-reID/bounding_box_train/0024_c1_f0052855.jpg',
    # ]
}

# for home, dirs, files in os.walk(IMAGES_SOURCE):
#     for filename in files:
#         class_id = int(filename[START_INDEX:END_INDEX])
#         if class_id not in samples.keys():
#             samples[class_id] = []
#         samples[class_id].append(os.path.join(home, filename))

models = {
    # '../configs/basis_exp/experiment_fedstil.yaml': r'MODEL_CHECKPOINTS_PATH',
    # '../configs/basis_exp/experiment_fedcurv.yaml': r'MODEL_CHECKPOINTS_PATH',
    # '../configs/basis_exp/experiment_fedweit.yaml': r'MODEL_CHECKPOINTS_PATH',
}

save_dir = './cam/'

if __name__ == '__main__':
    for method_config, model_ckpt in models.items():
        with open(method_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        method_name = config['exp_name']
        method_type = config['exp_method']
        model = parser_model(method_type, config['model_opts'])
        model.update_model(torch.load(model_ckpt))

        target_layers = [model.net.base.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

        for class_id, sample_paths in samples.items():
            for img_id, img_path in enumerate(sample_paths):
                rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
                rgb_img = np.float32(rgb_img) / 255

                input_tensor = transforms.ToTensor()(rgb_img).unsqueeze(0)
                grayscale_cam = cam(input_tensor, None)[0, :]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)
                cv2.imwrite(f'{save_dir}/{class_id}_{img_id}.jpg', np.uint8(255 * rgb_img[:, :, ::-1]))
                cv2.imwrite(f'{save_dir}/{class_id}_{img_id}_{method_name}.jpg', visualization)
