"""
Reference: RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
Code based on https://github.com/princeton-vl/RAFT/tree/master/core
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock
from utils.utils import coords_grid

from warp import apply_warp_by_field

from typing import List
from typing import Tuple

import pdb


def get_pos_encoding_table(n_position: int, d_hid: int):
    #  n_position, d_hid -- (1024, 256)

    # def get_position_angle_vec(position):
    #     # this part calculate the position In brackets
    #     return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    # encoding_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    # # [:, 0::2] are all even subscripts, is dim_2i
    # encoding_table[:, 0::2] = np.sin(encoding_table[:, 0::2])  # dim 2i
    # encoding_table[:, 1::2] = np.cos(encoding_table[:, 1::2])  # dim 2i+1

    # # encoding_table.shape -- (1024, 256)
    # return torch.FloatTensor(encoding_table).unsqueeze(0) # ==> [1, 1024, 256]


    encoding_table = torch.zeros(n_position, d_hid)
    index = torch.tensor([(2 * (j // 2) / d_hid) for j in range(d_hid)])
    for i in range(n_position):
        t = float(i)/torch.pow(10000, index)
        encoding_table[:, 0::2] = torch.sin(t[0::2])
        encoding_table[:, 1::2] = torch.cos(t[1::2])
    return encoding_table.unsqueeze(0)


class mask_RAFT(nn.Module):
    def __init__(self):
        super(mask_RAFT, self).__init__()

        class Args(object):
            def __init__(self):
                self.small = False
                self.corr_levels = 4
                self.corr_radius = 4                

        args = Args()
        self.args = args

        if args.small:  # False
            self.hidden_dim = 96
            self.context_dim = 64
            args.corr_radius = 3

        else:
            self.hidden_dim = 128
            self.context_dim = 128
            args.corr_radius = 4

        # feature network, context network, and update block
        if args.small:  # False
            self.fnet = SmallEncoder(output_dim=128, norm_fn="instance", dropout=0.0)
            self.cnet = SmallEncoder(output_dim=self.hidden_dim + self.context_dim, norm_fn="none", dropout=0.0)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=self.hidden_dim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", dropout=0.0)
            self.cnet = BasicEncoder(output_dim=self.hidden_dim + self.context_dim, norm_fn="batch", dropout=0.0) # Useless
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=self.hidden_dim)

        # self.pos_embed = get_pos_encoding_table(32*32,256).permute(0,2,1).view(1,-1,32,32)

        self.corr = CorrBlock(radius=self.args.corr_radius)

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(
        self,
        source_image,
        source_mask,
        target_mask,
        refine_time: int=12,
    ) -> List[torch.Tensor]:

        # run the feature network
        fmap1, fmap2 = self.fnet([source_mask, target_mask])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # Position Embedding
        pos_embed = (
            get_pos_encoding_table(fmap1.shape[2] * fmap1.shape[3], fmap1.shape[1])
            .permute(0, 2, 1)
            .view(1, -1, fmap1.shape[2], fmap1.shape[3])
        )
        fmap1 = fmap1 + pos_embed.to(fmap1.device)
        fmap2 = fmap2 + pos_embed.to(fmap2.device)

        # corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        net, inp = torch.split(fmap2, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(source_mask)

        flow_predictions = []
        # down_flow_predictions = []
        for i in range(refine_time):
            coords1 = coords1.detach()
            # corr = corr_fn(coords1)  # index correlation volume
            corr = self.corr(fmap1, fmap2, coords1)

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)
            # down_flow_predictions.append(coords1 - coords0)

        warped_image_mask_list: List[torch.Tensor] = []
        for flow_up in flow_predictions:
            image = apply_warp_by_field(source_image.clone(), flow_up)
            warped_image_mask_list.append(image)

        for flow_up in flow_predictions:
            mask = apply_warp_by_field(source_mask.clone(), flow_up)
            warped_image_mask_list.append(mask)

        return warped_image_mask_list

