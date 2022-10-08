"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 08日 星期四 01:39:22 CST
# ***
# ************************************************************************************/
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

import pdb


def coords_grid(batch: int, height: int, width: int):
    coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def gather(width: int, flattened_grid, y_coords, x_coords):
    linear_coordinates = y_coords * width + x_coords  # (H*W)
    linear_coordinates = linear_coordinates.long()  # torch.LongTensor(linear_coordinates.long())
    gathered_values = flattened_grid[linear_coordinates]
    return gathered_values


def interpolate_bilinear(grid, query_points, indexing: str = "ij"):
    """Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.

    Args:
      grid: a 3-D float `Tensor` of shape `[height, width, channels]`.
      query_points: a 2-D float `Tensor` of N points with shape `[N, 2]`.
      name: a name for the operation (optional).
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).

    Returns:
      values: a 2-D `Tensor` with shape `[N, channels]`
    """

    height, width, channels = grid.size()
    query_type = query_points.dtype
    grid_type = grid.dtype
    num_queries, _ = query_points.shape

    alphas = []
    floors = []
    ceils = []

    index_order = [0, 1] if indexing == "ij" else [1, 0]
    unstacked_query_points = torch.unbind(query_points, 1)

    for dim in index_order:
        queries = unstacked_query_points[dim]
        size_in_indexing_dimension = grid.size(dim)

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = size_in_indexing_dimension - 2  # query_type
        min_floor = 0.0  # query_type
        floor = torch.clamp(torch.clamp(torch.floor(queries), min=min_floor), max=max_floor)
        int_floor = floor.int()
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = queries - floor  # grid_type
        alpha = torch.clamp(alpha, min=0.0, max=1.0)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = alpha.unsqueeze(1)
        alphas.append(alpha)

    flattened_grid = grid.reshape([height * width, channels])

    # Grab the pixel values in the 4 corners around each query point
    top_left = gather(width, flattened_grid, floors[0], floors[1])  # (H*W, 3)
    top_right = gather(width, flattened_grid, floors[0], ceils[1])  # (H*W, 3)
    bottom_left = gather(width, flattened_grid, ceils[0], floors[1])  # (H*W, 3)
    bottom_right = gather(width, flattened_grid, ceils[0], ceils[1])  # (H*W, 3)

    interp_top = alphas[1].to(grid.device) * (top_right - top_left) + top_left
    interp_bottom = alphas[1].to(grid.device) * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0].to(grid.device) * (interp_bottom - interp_top) + interp_top

    return interp


def image_warp_by_field(img, warp_field):
    """Warps the giving image based on warp_field

    Args:
        img: [1, c, h, w] float `Tensor`
        warp_field: [1, 2, h, w] float `Tensor`

    Returns:
        interpolated: [1, c, h, w] float `Tensor`
    """

    _, _, height, width = img.size()

    flow = warp_field.clone().squeeze(dim=0).permute((1, 2, 0))

    grid_x, grid_y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="ij")
    stacked_grid = torch.stack([grid_x, grid_y], dim=2)  # (H, W, 2)
    stacked_grid = stacked_grid.float().to(img.device)

    query_points_on_grid = stacked_grid - flow  # (H, W, 2)
    query_points_flattened = query_points_on_grid.reshape([height * width, 2])  # (H*W, 2)

    img = img.reshape((3, height, width)).permute((1, 2, 0))
    interpolated = interpolate_bilinear(img, query_points_flattened, indexing="ij")  # (H*W, C)
    interpolated = interpolated.reshape((height, width, 3)).permute((2, 0, 1))
    interpolated = interpolated.reshape((1, 3, height, width))

    return interpolated


def apply_warp_by_field(im, warp_file):
    """
    im : [N,C,H,W] Tensor
    warp_file : [N,2,H,W]
    """
    new_im = []
    for i in range(im.shape[0]):
        new_im_each = image_warp_by_field(im[i : i + 1], warp_file[i : i + 1])
        new_im.append(new_im_each)
    return torch.cat(new_im, dim=0)


def get_pos_encoding_table(n_position: int, d_hid: int):
    #  n_position, d_hid -- (1024, 256)

    encoding_table = torch.zeros(n_position, d_hid)
    index = torch.tensor([(2 * (j // 2) / d_hid) for j in range(d_hid)])
    for i in range(n_position):
        t = float(i) / torch.pow(10000, index)
        encoding_table[:, 0::2] = torch.sin(t[0::2])
        encoding_table[:, 1::2] = torch.cos(t[1::2])
    return encoding_table.unsqueeze(0)


def bilinear_sampler(img, coords):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, align_corners=True)


class CorrBlock(nn.Module):
    def __init__(self, num_levels=4, radius=4):
        super(CorrBlock, self).__init__()

        self.num_levels = num_levels
        self.radius = radius

    def forward(self, fmap1, fmap2, coords):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        corr = corr / torch.sqrt(torch.tensor(dim).float())

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        corr_pyramid = []
        corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            corr_pyramid.append(corr)

        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=2).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = corr_pyramid[i]
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)

            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x: List[torch.Tensor]):
        batch_dim = x[0].shape[0]
        x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        # if self.training and self.dropout is not None:
        #     x = self.dropout(x)
        return torch.split(x, [batch_dim, batch_dim], dim=0)


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0)
        )

    def forward(self, net, inp, corr, flow) -> List[torch.Tensor]:
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class RAFT(nn.Module):
    """
    Reference: RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
    Code based on https://github.com/princeton-vl/RAFT/tree/master/core
    """

    def __init__(self):
        super(RAFT, self).__init__()

        class Args(object):
            def __init__(self):
                self.corr_levels = 4
                self.corr_radius = 4

        args = Args()
        self.args = args
        self.hidden_dim = 128
        self.context_dim = 128

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", dropout=0.0)
        self.cnet = BasicEncoder(output_dim=self.hidden_dim + self.context_dim, norm_fn="batch", dropout=0.0)  # Useless
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

    def forward(self, x):
        source_image = x[:, 0:3, :, :]
        source_mask = x[:, 3:6, :, :]
        target_mask = x[:, 6:9, :, :]

        # run the feature network
        fmap1, fmap2 = self.fnet([source_mask, target_mask])
        # fmap1.size()/fmap2.size() -- [1, 256, 32, 32]

        # Position Embedding
        pos_embed = (
            get_pos_encoding_table(fmap1.shape[2] * fmap1.shape[3], fmap1.shape[1])
            .permute(0, 2, 1)
            .view(1, -1, fmap1.shape[2], fmap1.shape[3])
        )
        # pos_embed.size() -- [1, 256, 32, 32]

        fmap1 = fmap1 + pos_embed.to(fmap1.device)
        fmap2 = fmap2 + pos_embed.to(fmap2.device)

        # run the context network
        net, inp = torch.split(fmap2, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        coords0, coords1 = self.initialize_flow(source_mask)

        flow_predictions = []

        refine_time: int = 6
        for i in range(refine_time):
            coords1 = coords1.detach()
            # corr = corr_fn(coords1)  # index correlation volume
            corr = self.corr(fmap1, fmap2, coords1)

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        warped_image_mask_list: List[torch.Tensor] = []
        for flow_up in flow_predictions:
            image = apply_warp_by_field(source_image.clone(), flow_up)
            warped_image_mask_list.append(image)

        for flow_up in flow_predictions:
            mask = apply_warp_by_field(source_mask.clone(), flow_up)
            warped_image_mask_list.append(mask)

        warped_source_images = warped_image_mask_list[0:refine_time]
        warped_source_masks = warped_image_mask_list[refine_time:]

        return torch.cat(
            [source_mask, target_mask] + warped_source_masks + [source_image, target_mask] + warped_source_images,
            dim=0,
        )
