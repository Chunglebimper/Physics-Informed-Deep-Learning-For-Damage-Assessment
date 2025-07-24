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
        #self.stitch_list = []

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

            #for each top left pixel in each patch
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):

                    #creates a patch from the mask image
                    patch = mask[y:y + patch_size, x:x + patch_size]

                    #determines whether or not it's included in the training dataset
                    include = (4 in patch or 3 in patch or 2 in patch or np.random.rand() < float(class0and1percent*0.01))


                    if include:

                        #opens the post disaster png file and turns it into a 3 band numpy array
                        path = os.path.join(post_dir, f'{basename}_post_disaster.png')
                        img = Image.open(path)
                        file_array = np.transpose(np.array(img), (2, 0, 1))

                        #initializes an empty array patch
                        array_patch = [[], self.patch_size, self.patch_size]

                        #creates an RGB patch and puts it into the initialized array_patch
                        for band in range(3):
                            array_patch[band] = file_array[band][y:y + patch_size, x:x + patch_size]

                        #checks to see if there are 10 or more 0,0,0 black pixels in the patch
                        #if there are 10, then the patch data is put into a delete_list
                        c = 0
                        for row in range(len(array_patch[0])):
                            for col in range(len(array_patch[0][row])):
                                if array_patch[0][row][col] == array_patch[1][row][col] == array_patch[2][row][col] == 0:
                                    c += 1
                                    if c == 10:
                                        print(basename, x, y, 'post', 'removed black patch')
                                        self.delete_list.append([basename, x, y])
                                        break

                        #this is duplicated code for the pre-patches
                        path = os.path.join(pre_dir, f'{basename}_pre_disaster.png')
                        img = Image.open(path)
                        file_array = np.transpose(np.array(img), (2, 0, 1))
                        array_patch = [[], self.patch_size, self.patch_size]
                        for band in range(3):
                            array_patch[band] = file_array[band][y:y + patch_size, x:x + patch_size]
                        c = 0
                        for row in range(len(array_patch[0])):
                            for col in range(len(array_patch[0][row])):
                                if array_patch[0][row][col] == array_patch[1][row][col] == array_patch[2][row][
                                    col] == 0:
                                    c += 1
                                    if c == 10:
                                        print(basename, x, y, 'pre', 'removed black patch')
                                        self.delete_list.append([basename, x, y])
                                        break

                        #if the patch is an included patch dataset isn't in the delete_list (doesn't have 10 or more black pixels)
                        #note that if either the pre or post patches have black spots, then neither will be used
                        if not [basename, x, y] in self.delete_list:

                            #calls it a priority image if it has a high damage class pixel
                            is_priority = any(cls in patch for cls in [2, 3, 4])
                            print(f'Patch featuring class 4: \t{basename, x, y}\n' if 4 in patch else "", end="")

                            #it then appends the data to the samples list which the model uses to train
                            #notes: neither these patches nor their pre or post counterpart have 10 or more black pixels
                            self.samples.append((basename, x, y, is_priority))



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

        #initializes array
        array = np.zeros((512,512)).astype(np.uint8)
        image_number = f'000{image_number}'

        #creates the border width that we're taking off of each side of the patch
        border = (self.patch_size - self.stride) // 2

        for number, patch in enumerate(self.samples):

            #checks to see if the patch in question contains the image_number at the end of the basename
            if patch[0][-len(str(image_number)):] == str(image_number):
                print(patch, 'stitched to full_mask_image')

                #initializing the coordinates of the top left hand corner of the patch
                x = patch[1]
                y = patch[2]

                #calls in the rgb patches from the indexed class
                pre, post, _, _ = self[number]

                #generates the mask image from the pre- and post-patches
                with torch.no_grad():
                    damage_out = model(pre, post)
                    pred = torch.argmax(damage_out.squeeze(), dim=0).cpu().numpy().astype(np.uint8)
                    print(np.shape(pred))

                #this removes the borders of the mask patch and appends it to the array
                #if there is no data the array defaults to 0
                #the borders are removed to deal with stride overlap
                #the borders of the entire image are also discarded
                #you can do casework to eliminate this issue
                array[y + border: y + self.patch_size - border, x + border: x + self.patch_size - border]\
                    = pred[border: self.patch_size - border, border: self.patch_size - border]


        #normalization for display
        array = (array * 255 / 4).astype(np.uint8)
        img = Image.fromarray(array, 'L')
        img.save(f'{results_path}/full_mask_image_{image_number}.png')
        print('\nSaved stitched mask image!\n')


