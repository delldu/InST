"""Image/Video Shape Style Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 08日 星期四 01:39:22 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

import todos
from . import shape_style

import pdb


def get_model():
    """Create model."""

    model_path = "models/image_shape_style.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = shape_style.RAFT()
    todos.model.load(model, checkpoint)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/image_shape_style.torch"):
        model.save("output/image_shape_style.torch")

    return model, device


def image_predict(source_images, source_masks, target_masks, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    # load files
    source_image_files = todos.data.load_files(source_images)
    source_mask_files = todos.data.load_files(source_masks)
    target_mask_files = todos.data.load_files(target_masks)

    # start predict
    progress_bar = tqdm(total=len(source_image_files) * len(target_mask_files))
    for i, source_image_file in enumerate(source_image_files):
        source_image_tensor = todos.data.load_tensor(source_image_file)
        source_image_tensor = F.interpolate(source_image_tensor, size=(256, 256), mode="bilinear", align_corners=False)

        source_mask_tensor = todos.data.load_tensor(source_mask_files[i])
        source_mask_tensor = F.interpolate(source_mask_tensor, size=(256, 256), mode="bilinear", align_corners=False)

        for target_mask_file in target_mask_files:
            progress_bar.update(1)

            target_mask_tensor = todos.data.load_tensor(target_mask_file)
            target_mask_tensor = F.interpolate(
                target_mask_tensor, size=(256, 256), mode="bilinear", align_corners=False
            )

            # input == source_image + source_mask + target_mask
            input_tensor = torch.cat((source_image_tensor, source_mask_tensor, target_mask_tensor), dim=1)
            output_tensor = todos.model.forward(model, device, input_tensor)

            output_file = f"{output_dir}/{i+1:02d}_{os.path.basename(target_mask_file)}"
            todos.data.save_tensor([output_tensor], output_file)

    todos.model.reset_device()
