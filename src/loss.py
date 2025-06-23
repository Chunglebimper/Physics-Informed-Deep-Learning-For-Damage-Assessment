import numpy as np
import torch

# GLCM-based texture loss: encourages model to capture texture-based patterns
def calculate_glcm_energy(patch, levels):
    if patch.size == 0:
        return 0.0
    patch = np.clip(patch, 0, 255).astype(np.uint8)
    patch = (patch * (levels - 1) / 255.0).astype(np.uint8)
    glcm = np.zeros((levels, levels), dtype=np.float32)
    #print("patch.shape[0] = ", patch.shape[0])
    #print("glcm before = ",glcm)
    for y in range(patch.shape[0] - 1):
        for x in range(patch.shape[1] - 1):
            glcm[patch[y,x], patch[y,x+1]] += 1     # right
            glcm[patch[y,x], patch[y+1,x]] += 1     # up
            glcm[patch[y,x], patch[y+1,x+1]] += 1   # top-left

            #glcm[patch[y, x], patch[y, x + 2]] += 1  # right 2
            #glcm[patch[y, x], patch[y + 2, x]] += 1  # up 2
            #glcm[patch[y, x], patch[y + 2, x + 1]] += 1  # top-left 2

    glcm /= glcm.sum() + 1e-6
    #print("glcm after = ", glcm)
    return np.sum(glcm ** 2)

def adaptive_texture_loss(pre_img, post_img, pred_classes, sample_size, levels, debris_class=4, patch_size=32):
    device = post_img.device
    delta = torch.abs(post_img - pre_img)
    delta_gray = (0.2989 * delta[:, 0] +
                  0.5870 * delta[:, 1] +
                  0.1140 * delta[:, 2]) # what are the other values here applied to the delta RGB values?
    loss = torch.tensor(0.0, device=device)
    count = 0
    for b in range(delta_gray.shape[0]):
        mask = (pred_classes[b] == debris_class)
        if not torch.any(mask): continue
        img_np = (delta_gray[b].cpu().numpy() * 255).astype(np.uint8)
        coords = torch.nonzero(mask)
        if len(coords) > sample_size:
            coords = coords[torch.randperm(len(coords))[:sample_size]]
        for y, x in coords:
            y, x = y.item(), x.item()
            patch = img_np[max(0,y-patch_size//2):min(img_np.shape[0],y+patch_size//2),
                           max(0,x-patch_size//2):min(img_np.shape[1],x+patch_size//2)]
            energy = calculate_glcm_energy(patch, levels)
            weight = 1.0 if energy < 0.7 else 0.3
            loss += weight * (1.0 - energy)
            count += 1
    return loss / (count + 1e-6)
