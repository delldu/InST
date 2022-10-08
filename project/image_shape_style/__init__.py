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
from torchvision.utils import save_image

import redos
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


def model_forward(model, device, input_tensor):
    return todos.model.forward(model, device, input_tensor)


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.shape_style(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, host="localhost", port=6379):
    # load model
    model, device = get_model()

    def do_service(input_file, output_file, targ):
        print(f"  shape_style {input_file} ...")
        try:
            content_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, content_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_shape_style", do_service, host, port)


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
            output_tensor = model_forward(model, device, input_tensor)

            output_file = f"{output_dir}/{i+1:02d}_{os.path.basename(target_mask_file)}"
            save_image(
                output_tensor.cpu(),
                output_file,
                nrow=2 + 6,  # 6 -- refine_time defined in model
                padding=2,
                pad_value=0.5,
            )


def video_service(input_file, output_file, targ):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    print(f"  shape_style {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def photo_style_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        content_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        content_tensor = content_tensor[:, 0:3, :, :]
        output_tensor = model_forward(model, device, content_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=photo_style_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.shape_style(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, host="localhost", port=6379):
    return redos.video.service(name, "video_photo_style", video_service, host, port)
