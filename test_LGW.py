import argparse
import sys
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image

sys.path.append("mask_RAFT")
from mask_RAFT.mask_raft import mask_RAFT
from segmentation import segment

import pdb


def get_args():
    parser = argparse.ArgumentParser("Test Warp Model for fast warpping")
    # basic options
    parser.add_argument("--cpu", action="store_true", help="wheather to use cpu , if not set, use gpu")

    # data options
    parser.add_argument("--source_path", type=str, help="path of source image")
    parser.add_argument("--source_dir", type=str, help="Directory path to source image")
    parser.add_argument(
        "--source_mask_dir", type=str, help="Directory path to source mask images, if None, use PointRender to segment"
    )
    parser.add_argument(
        "--source_mask_path", type=str, help="path of source mask image, if None, use PointRender to segment"
    )
    parser.add_argument("--target_dir", type=str, help="Directory path to target images")
    parser.add_argument("--target_path", type=str, help="path of target images")
    parser.add_argument(
        "--target_mask_dir", type=str, help="Directory path to target mask images, if None, use PointRender to segment"
    )
    parser.add_argument(
        "--target_mask_path", type=str, help="path of target mask images, if None, use PointRender to segment"
    )
    parser.add_argument("--im_height", type=int, default=256)
    parser.add_argument("--im_width", type=int, default=256)

    # model options
    parser.add_argument("--checkpoint", type=str, required=True, help="module of warpping")
    parser.add_argument("--refine_time", type=int, default=3, help="warp refine time")

    # other options
    parser.add_argument("--output_dir", type=str, default="./output")

    args = parser.parse_args()
    return args


def _get_transform(size=None):
    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def test(args):
    device = torch.device("cuda" if not args.cpu else "cpu")

    if args.source_path or args.source_dir:
        if args.source_path:
            source_paths = [Path(args.source_path)]
        else:
            source_dir = Path(args.source_dir)
            source_paths = sorted(
                [f for f in source_dir.glob("*.jpg")]
                + [f for f in source_dir.glob("*.png")]
                + [f for f in source_dir.glob("*.jpeg")]
            )

    if args.source_mask_path:
        source_mask_paths = [Path(args.source_mask_path)]
    elif args.source_mask_dir:
        source_mask_dir = Path(args.source_mask_dir)
        source_mask_paths = sorted(
            [f for f in source_mask_dir.glob("*.jpg")]
            + [f for f in source_mask_dir.glob("*.png")]
            + [f for f in source_mask_dir.glob("*.jpeg")]
        )
    else:
        # Use PointRend to help segment
        source_mask_paths = segment(source_paths)

    if args.target_path or args.target_dir:
        if args.target_path:
            target_paths = [Path(args.target_path)]
        else:
            target_dir = Path(args.target_dir)
            target_paths = sorted(
                [f for f in target_dir.glob("*.jpg")]
                + [f for f in target_dir.glob("*.png")]
                + [f for f in target_dir.glob("*.jpeg")]
            )

    if args.target_mask_path:
        target_mask_paths = [Path(args.target_mask_path)]
    elif args.target_mask_dir:
        target_mask_dir = Path(args.target_mask_dir)
        target_mask_paths = sorted(
            [f for f in target_mask_dir.glob("*.jpg")]
            + [f for f in target_mask_dir.glob("*.png")]
            + [f for f in target_mask_dir.glob("*.jpeg")]
        )
    else:
        assert args.target_path or args.target_dir
        logger.info("Using PointRender to segment target images...")
        target_mask_paths = segment(target_paths)


    if len(source_mask_paths) == 0 or len(target_mask_paths) == 0:
        logger.warn("no test images")
        return

    # Model
    model = mask_RAFT().to(device)

    checkpoint = torch.load(args.checkpoint)
    # model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
    model.load_state_dict(checkpoint["modelstate"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    source_transform = _get_transform(size=(args.im_height, args.im_width))
    target_transform = _get_transform(size=(args.im_height, args.im_width))

    model.eval()
    # model -- mask_RAFT(...)
    torch.jit.script(model)

    for i, source_mask_path in enumerate(source_mask_paths):
        source_mask_img = Image.open(source_mask_path).convert("RGB")
        source_mask = source_transform(source_mask_img).unsqueeze(0)
        # source_transform --
        # Compose(
        #     Resize(size=(256, 256), interpolation=bilinear, max_size=None, antialias=None)
        #     ToTensor()
        # )

        source_img = Image.open(source_paths[i]).convert("RGB")
        source_image = source_transform(source_img).unsqueeze(0).to(device)

        source_mask = source_mask.to(device)
        for target_mask_path in target_mask_paths:
            target_mask_img = Image.open(target_mask_path).convert("RGB")
            target_mask = target_transform(target_mask_img).unsqueeze(0).to(device)
            # target_transform --
            # Compose(
            #     Resize(size=(256, 256), interpolation=bilinear, max_size=None, antialias=None)
            #     ToTensor()
            # )
            with torch.no_grad():
                warped_image_mask_list = model(source_image, source_mask, target_mask, refine_time=args.refine_time)

            warped_source_images = warped_image_mask_list[0 : args.refine_time]
            warped_source_masks = warped_image_mask_list[args.refine_time: ]

            save_image_name = (
                source_mask_path.name.split(".")[0]
                + "_shaped_"
                + target_mask_path.name.split(".")[0]
                + f"_refine{args.refine_time}.jpg"
            )
            save_image(
                torch.cat(
                    [source_mask, target_mask]
                    + warped_source_masks
                    + [source_image, target_mask]
                    + warped_source_images,
                    dim=0,
                ).cpu(),
                str(Path(output_dir, save_image_name)),
                scale_each=True,
                nrow=2 + args.refine_time,
                padding=2,
                pad_value=255,
            )
            single_image_name = (
                f"single_"
                + source_mask_path.name.split(".")[0]
                + "_shaped_"
                + target_mask_path.name.split(".")[0]
                + f"_refine{args.refine_time}.jpg"
            )
            save_image(
                warped_source_images[-1].cpu(),
                str(Path(output_dir, single_image_name)),
                scale_each=True,
                nrow=1,
                padding=2,
                pad_value=255,
            )


if __name__ == "__main__":
    args = get_args()
    test(args)
