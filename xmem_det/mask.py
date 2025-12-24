import torch
import torch.nn as nn
import torch.nn.functional as F

# Add this method to your TemporalPointPillar class:

def transform_mask(self, mask_prev, T_rel, H, W):
    """
    Transform detection masks from previous frame to current frame using ego-motion.
    
    Args:
        mask_prev: [B, N_obj, H, W] detection masks from previous frame
        T_rel: [3, 3] or [4, 4] transformation matrix from prev frame to current frame
        H, W: BEV grid dimensions
    
    Returns:
        mask_transformed: [B, N_obj, H, W] masks warped to current frame coordinates
    """
    if T_rel is None or mask_prev is None:
        return mask_prev
    
    B, N_obj = mask_prev.shape[0], mask_prev.shape[1]
    device = mask_prev.device
    dtype = mask_prev.dtype
    
    # Extract 2D transformation (rotation + translation)
    T = T_rel.unsqueeze(0) if T_rel.dim() == 2 else T_rel
    
    if T.size(-1) == 4:
        # 4x4 matrix
        r11, r12 = T[:, 0, 0], T[:, 0, 1]
        r21, r22 = T[:, 1, 0], T[:, 1, 1]
        tx, ty = T[:, 0, 3], T[:, 1, 3]
    else:
        # 3x3 matrix
        r11, r12 = T[:, 0, 0], T[:, 0, 1]
        r21, r22 = T[:, 1, 0], T[:, 1, 1]
        tx, ty = T[:, 0, 2], T[:, 1, 2]
    
    # Normalize translation to grid coordinates [-1, 1]
    x_min, y_min, _, x_max, y_max, _ = self.pc_range
    range_x = float(x_max - x_min)
    range_y = float(y_max - y_min)
    
    # Normalized translation (in [-1, 1] range for grid_sample)
    tx_norm = 2.0 * tx / range_x
    ty_norm = 2.0 * ty / range_y
    
    # Build affine transformation matrix for grid_sample
    # Note: grid_sample expects [B, 2, 3] affine matrix
    # Format: [[r11, r12, tx], [r21, r22, ty]]
    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = r11
    theta[:, 0, 1] = r12
    theta[:, 0, 2] = tx_norm
    theta[:, 1, 0] = r21
    theta[:, 1, 1] = r22
    theta[:, 1, 2] = ty_norm
    
    # Create sampling grid
    grid = F.affine_grid(theta, size=(B, 1, H, W), align_corners=False)
    
    # Transform each object mask
    masks_transformed = []
    for obj_idx in range(N_obj):
        mask_single = mask_prev[:, obj_idx:obj_idx+1, :, :]  # [B, 1, H, W]
        
        # Apply transformation using bilinear interpolation
        mask_warped = F.grid_sample(
            mask_single,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        masks_transformed.append(mask_warped)
    
    # Stack all object masks
    mask_transformed = torch.cat(masks_transformed, dim=1)  # [B, N_obj, H, W]
    
    return mask_transformed

