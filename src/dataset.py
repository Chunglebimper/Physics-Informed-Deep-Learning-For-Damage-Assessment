import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

# Dataset class to handle pre, post, and mask images
class DamageDataset(Dataset):
    def __init__(self, pre_dir, post_dir, mask_dir, class0and1percent=10, patch_size=128, stride=64, mode='post'):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.delete_list = []
        self.stitch_list = []

        # Standard transforms
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Collect image samples
        self.filenames = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(f"_{mode}_disaster_target.png")])
        self.samples = []
        for fname in self.filenames:
            basename = fname.replace(f"_{mode}_disaster_target.png", "")
            mask = np.array(Image.open(os.path.join(mask_dir, fname)).convert('L'))
            h, w = mask.shape

            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = mask[y:y + patch_size, x:x + patch_size]
                    include = (4 in patch or 3 in patch or 2 in patch or np.random.rand() < float(class0and1percent*0.1))
                    if include:
                        path = os.path.join(post_dir, f'{basename}_post_disaster.png')
                        img = Image.open(path)
                        file_array = np.transpose(np.array(img), (2, 0, 1))
                        array_patch = [[], h, w]
                        for band in range(3):
                            array_patch[band] = file_array[band][y:y + patch_size, x:x + patch_size]
                        c = 0
                        for row in range(len(array_patch[0])):
                            for col in range(len(array_patch[0][row])):
                                if array_patch[0][row][col] == array_patch[1][row][col] == array_patch[2][row][col] == 0:
                                    c += 1
                                    if c == 10:
                                        print(fname, x, y, 'post')
                                        self.delete_list.append([basename, x, y])
                                        break

                        path = os.path.join(pre_dir, f'{basename}_pre_disaster.png')
                        img = Image.open(path)
                        file_array = np.transpose(np.array(img), (2, 0, 1))
                        array_patch = [[], h, w]
                        for band in range(3):
                            array_patch[band] = file_array[band][y:y + patch_size, x:x + patch_size]
                        c = 0
                        for row in range(len(array_patch[0])):
                            for col in range(len(array_patch[0][row])):
                                if array_patch[0][row][col] == array_patch[1][row][col] == array_patch[2][row][
                                    col] == 0:
                                    c += 1
                                    if c == 10:
                                        print(fname, x, y, 'pre')
                                        self.delete_list.append([basename, x, y])
                                        break

                        if not [basename, x, y] in self.delete_list:
                            is_priority = any(cls in patch for cls in [2, 3, 4])
                            print(f'Patch featuring class 4: \t{basename, x, y}\n' if 4 in patch else "", end="")
                            self.samples.append((basename, x, y, is_priority))
                            if x % patch_size == 0 and y % patch_size == 0:
                                self.stitch_list.append([basename, x, y])



    def __len__(self):
        return len(self.samples)




    def __getitem__(self, idx):
        basename, x, y, is_priority = self.samples[idx]
        pre_img = np.array(Image.open(os.path.join(self.pre_dir, f"{basename}_pre_disaster.png")).convert('RGB'))
        post_img = np.array(Image.open(os.path.join(self.post_dir, f"{basename}_post_disaster.png")).convert('RGB'))
        mask = np.array(Image.open(os.path.join(self.mask_dir, f"{basename}_{self.mode}_disaster_target.png")).convert('L'))

        # Crop to patch
        pre_patch = pre_img[y:y + self.patch_size, x:x + self.patch_size]
        post_patch = post_img[y:y + self.patch_size, x:x + self.patch_size]
        mask_patch = mask[y:y + self.patch_size, x:x + self.patch_size]

        # Apply transforms
        transform = self.aug_transform if is_priority else self.base_transform
        pre_patch = transform(Image.fromarray(pre_patch))
        post_patch = transform(Image.fromarray(post_patch))

        return pre_patch, post_patch, torch.from_numpy(mask_patch).long(), f"{basename}_x{x}_y{y}"

    def create_stitched_mask_image(self, image_number, model, results_path):
        '''
        :param image_number: image number with leading 0s ex. ---> 00003
        :param model: this is an initiated class instance
        :param results_path: preset in the mkdirs class
        :return: creates a stitched mask image which is added to the log directory
        '''
        array = np.zeros((512,512)).astype(np.uint8)
        image_number = f'000{image_number}'
        for number, patch in enumerate(self.stitch_list):
            if patch[0][-len(str(image_number)):] == str(image_number):
                print(patch, 'yay')
                x = patch[1]
                y = patch[2]
                pre, post, _, _ = self[number]
                with torch.no_grad():
                    damage_out = model(pre, post)
                    pred = torch.argmax(damage_out.squeeze(), dim=0).cpu().numpy().astype(np.uint8)
                array[y : y + self.patch_size, x : x + self.patch_size] = pred


        array = (array * 255 / 4).astype(np.uint8)
        img = Image.fromarray(array, 'L')
        img.save(f'{results_path}/full_mask_image_{image_number}.png')


