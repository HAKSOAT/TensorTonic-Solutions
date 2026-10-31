import torch
import torch.nn.functional as F

def vision_transformer_patchify(
    images: torch.Tensor, patch_height: int, patch_width: int,
) -> dict:
    """
    Returns a dict: tokens (image dtype), coordinates (int64 tensor).
    """
    assert len(images.shape) == 4
    batch, channels, height, width = images.shape # batch, channels, height, width
    assert height % patch_height == 0
    assert width % patch_width == 0
    ksize = (patch_height, patch_width)
    # Convolution patch style extraction
    tokens = F.unfold(images, kernel_size=ksize, stride=ksize) # batch, patch_dim, num_patches
    tokens = tokens.permute(0, 2, 1) # batch, num_patches, patch_dim
    patch_count = (height // patch_height, width // patch_width) # per channel
    # Creating coordinates for the patch grids
    mask = torch.ones(patch_count[0], patch_count[1])
    coordinates = torch.nonzero(mask)
    return {"tokens": tokens, "coordinates": coordinates}
