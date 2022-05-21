# Helper functions for differentiable image warping

# Code based on https://github.com/seasonSH/WarpGAN

import torch
# import numpy as np
import pdb

# This wraps array_ops.gather. We reshape the image data such that the
# batch, y, and x coordinates are pulled into the first dimension.
# Then we gather. Finally, we reshape the output back. It's possible this
# code would be made simpler by using array_ops.gather_nd.
def gather(width:int, flattened_grid, y_coords, x_coords, name: str):
    linear_coordinates = y_coords * width + x_coords  # (H*W)
    linear_coordinates = linear_coordinates.long()  # torch.LongTensor(linear_coordinates.long())
    gathered_values = flattened_grid[linear_coordinates]
    return gathered_values


def interpolate_bilinear(grid, query_points, indexing:str="ij"):
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

    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the inputs
        invalid.
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
        min_alpha = 0.0  # grid_type
        max_alpha = 1.0  # grid_type
        alpha = torch.clamp(torch.clamp(alpha, min=min_alpha), max=max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = alpha.unsqueeze(1)
        alphas.append(alpha)

    flattened_grid = grid.reshape([height * width, channels])

    # Grab the pixel values in the 4 corners around each query point
    top_left = gather(width, flattened_grid, floors[0], floors[1], "top_left")  # (H*W, 3)
    top_right = gather(width, flattened_grid, floors[0], ceils[1], "top_right")  # (H*W, 3)
    bottom_left = gather(width, flattened_grid, ceils[0], floors[1], "bottom_left")  # (H*W, 3)
    bottom_right = gather(width, flattened_grid, ceils[0], ceils[1], "bottom_right")  # (H*W, 3)

    # Now do the actual interpolation
    # try:
    #     interp_top = alphas[1].cuda() * (top_right - top_left) + top_left
    #     interp_bottom = alphas[1].cuda() * (bottom_right - bottom_left) + bottom_left
    #     interp = alphas[0].cuda() * (interp_bottom - interp_top) + interp_top

    # except:
    #     interp_top = alphas[1] * (top_right - top_left) + top_left
    #     interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    #     interp = alphas[0] * (interp_bottom - interp_top) + interp_top

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


    # grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    # stacked_grid = np.stack([grid_y, grid_x], axis=2)  # (H, W, 2)
    # stacked_grid = torch.from_numpy(stacked_grid).float().to(img.device)
    grid_x, grid_y = torch.meshgrid(torch.arange(width), torch.arange(height))
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
    new_im = torch.cat(new_im, dim=0)

    return new_im
