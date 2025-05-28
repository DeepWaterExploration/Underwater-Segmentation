"""SUIM Dataloader"""
import os
import random
import numpy as np
import torch
import torch.utils.data as data

import sys
sys.path.append(".")

from PIL import Image, ImageOps, ImageFilter
from py_tools.display_tools import display, display_multiple

__all__ = ['suimdataset']


class SUIMDataset(data.Dataset):

    BASE_DIR = 'SUIM'
    NUM_CLASS = 2  # 0 for water, 1 for not-water

    def __init__(self, root='./data/SUIM', split='train', mode=None, transform=None,
                 base_size=520, crop_size=480, **kwargs):
        super(SUIMDataset, self).__init__()
        self.root = root
        self.split = split
        # If mode is not specified, use the split (e.g., 'train', 'test')
        # This mode is used in __getitem__ to determine behavior (e.g., augmentations, return values)
        self.mode = mode if mode is not None else split
        self.transform = transform # General transform for the image (e.g., ToTensor, Normalize)
        self.base_size = base_size
        self.crop_size = crop_size

        self.images, self.mask_paths = _get_suim_image_mask_paths(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths)), \
            f"Number of images and masks do not match: {len(self.images)} images, {len(self.mask_paths)} masks for split {self.split}"
        if len(self.images) == 0:
            raise RuntimeError(f"Found 0 images in subfolders of: {self.root}/{self.split}\n"
                               "Please check your dataset path and structure.")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')

        if self.mode == 'test':
            # For test mode, typically only the image and its name are needed
            # The pre-supplied transform is usually for images (e.g. ToTensor, Normalize)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])

        mask = Image.open(self.mask_paths[index]).convert('L') # Ensure mask is loaded as grayscale

        # Synchronized transform for image and mask
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val': # Add 'val' to your data directory if you use this mode
            img, mask = self._val_sync_transform(img, mask)
        elif self.mode == 'testval': # Used if evaluating on a test set with labels
                                     # Or if you want to apply specific non-augmented transforms
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # If mode is something else (e.g. 'test' but labels are available and needed),
        # you might need to adjust logic or ensure mode is 'testval'.
        # For now, 'test' mode strictly follows the original logic of returning only image.

        # General image transform (applied after augmentations or specific transforms)
        if self.transform is not None:
            img = self.transform(img) # Usually ToTensor and Normalize

        return img, mask # Mask is already transformed to Tensor in _sync_transform or _val_sync_transform

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # Center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # Final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # Random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # Random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # Pad crop
        if short_size < crop_size:
            # Pad if image is smaller than crop_size after scaling
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            # Pad right and bottom. Fill with 0 (water).
            # If you want to pad with "not water" (class 1), you'd need to handle this carefully,
            # as ImageOps.expand fills with a single value.
            # For segmentation, often an ignore_index is used for padding if the loss supports it.
            # Here, we fill image with 0 (black) and mask with 0 (water).
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0) # Pad mask with 0 (water)

        # Random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # Gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # Final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        # Converts PIL image to numpy array.
        # Further transforms like ToTensor and Normalize are typically done by self.transform
        return np.array(img)

    def _mask_transform(self, mask_pil_image):
        # Converts PIL mask (L mode) to PyTorch tensor with class indices
        target = np.array(mask_pil_image).astype('int32')
        # Assuming water is 0 in the BMP. All other values are considered "not water" (class 1).
        target[target != 0] = 1
        return torch.LongTensor(target)

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    # pred_offset might be relevant if your model's output does not start from class 0.
    # For binary 0/1, this is typically 0.
    @property
    def pred_offset(self):
        return 0


def _get_suim_image_mask_paths(dataset_root, split):
    img_paths = []
    mask_paths = []
    
    img_folder = os.path.join(dataset_root, split, 'images')
    mask_folder = os.path.join(dataset_root, split, 'masks')

    if not os.path.isdir(img_folder):
        print(f"Image folder not found: {img_folder}")
        return img_paths, mask_paths
    if not os.path.isdir(mask_folder) and split != 'test': # Masks might be optional for a pure test set
        print(f"Mask folder not found: {mask_folder}")
        # Decide if this is an error or if masks can be optional for this split
        # For now, we'll try to find masks, and __init__ will assert their lengths match.

    for filename in os.listdir(img_folder):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(img_folder, filename)
            
            # Construct corresponding mask name and path
            mask_filename = filename.rsplit('.', 1)[0] + '.bmp' # Handles names like "img.a.jpg" -> "img.a.bmp"
            mask_path = os.path.join(mask_folder, mask_filename)

            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                img_paths.append(img_path)
                mask_paths.append(mask_path)
            elif os.path.isfile(img_path) and split == 'test' and not os.path.isfile(mask_path):
                # If it's the test set and mask is not found, some users might want to
                # still include the image and a dummy/None mask path.
                # The current __getitem__ for mode=='test' doesn't use mask_path.
                # However, the assertion len(images)==len(mask_paths) in __init__ will fail.
                # For consistency and to satisfy the assertion, if test masks are expected (even if not used by __getitem__ for 'test' mode),
                # this pair should be skipped or handled.
                # If test masks are truly optional, the logic in __init__ and here needs adjustment.
                # For now, we strictly require pairs.
                print(f"Image found but mask missing (or vice-versa): {img_path} / {mask_path}")
            else:
                 if not os.path.isfile(img_path):
                     print(f"Image file check failed (already listed but not found?): {img_path}")
                 if not os.path.isfile(mask_path):
                     print(f"Mask file not found for image {img_path}: {mask_path}")


    if not img_paths:
        print(f"No image files with corresponding masks found in {img_folder} and {mask_folder}")
    else:
        print(f"Found {len(img_paths)} image-mask pairs in {dataset_root}/{split}")
        
    return img_paths, mask_paths


if __name__ == '__main__':
    # Example of how to use the SUIMDataset
    # 1. First, ensure you have the './data/SUIM' directory structured as expected:
    #    data/SUIM/train/images/*.jpg
    #    data/SUIM/train/masks/*.bmp
    #    data/SUIM/test/images/*.jpg (masks optional or required depending on your test case)
    #    data/SUIM/test/masks/*.bmp

    # Create a dummy dataset for testing if you don't have the full one yet
    # os.makedirs('./data/SUIM/train/images', exist_ok=True)
    # os.makedirs('./data/SUIM/train/masks', exist_ok=True)
    # Image.new('RGB', (60, 30), color = 'red').save('./data/SUIM/train/images/sample1.jpg')
    # Image.new('L', (60, 30), color = 0).save('./data/SUIM/train/masks/sample1.bmp') # Water
    # Image.new('RGB', (60, 30), color = 'blue').save('./data/SUIM/train/images/sample2.jpg')
    # Image.new('L', (60, 30), color = 255).save('./data/SUIM/train/masks/sample2.bmp') # Not water

    print("Attempting to load 'train' split...")
    try:
        # For the transformations passed to `transform` in `__init__`:
        # These are typically `transforms.ToTensor()` and `transforms.Normalize()`
        # from `torchvision.transforms`.
        # For simplicity in this example, we'll pass None, so images will be numpy arrays.
        # In a real scenario:
        # import torchvision.transforms as T
        # image_transforms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        train_dataset = SUIMDataset(root='./data/SUIM', split='train', mode='train', transform=None, base_size=256, crop_size=224)
        
        if len(train_dataset) > 0:
            img, label = train_dataset[0] # Get the first sample
            print(f"\n--- Train Sample 0 ---")
            print(f"Image type: {type(img)}, Image shape: {img.shape if isinstance(img, np.ndarray) else 'PIL Image'}")
            print(f"Label type: {type(label)}, Label shape: {label.shape}, Label dtype: {label.dtype}")
            print(f"Unique values in label: {torch.unique(label) if isinstance(label, torch.Tensor) else np.unique(label)}")
            # Expected unique values for label: tensor([0]) or tensor([1]) or tensor([0, 1])

            # Example of using DataLoader
            train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
            for i, (imgs, labels) in enumerate(train_loader):
                print(f"\nBatch {i+1}:")
                print(f"Images batch shape: {imgs.shape}, type: {imgs.dtype}")
                print(f"Labels batch shape: {labels.shape}, type: {labels.dtype}")
                print(f"Unique values in labels batch: {torch.unique(labels)}")
                # if i == 0: # Just show first batch
                    # print(labels.shape)
                display_multiple([labels, imgs], ["mask", "image"])
        else:
            print("Train dataset is empty. Please check paths and file availability.")

    except RuntimeError as e:
        print(f"Error initializing or using dataset: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure your dataset is correctly placed and paths are correct.")

    print("\nAttempting to load 'test' split...")
    try:
        # For test mode, masks are not returned by __getitem__
        test_dataset = SUIMDataset(root='./data/SUIM', split='test', mode='test', transform=None)
        if len(test_dataset) > 0:
            img, filename = test_dataset[0]
            print(f"\n--- Test Sample 0 ---")
            print(f"Image type: {type(img)}, Image shape: {img.shape if isinstance(img, np.ndarray) else 'PIL Image'}")
            print(f"Filename: {filename}")
        else:
            print("Test dataset could not be loaded or is empty. Ensure `data/SUIM/test/images` and `data/SUIM/test/masks` (if using `mode='testval'`) exist and contain paired files.")

    except RuntimeError as e:
        print(f"Error initializing or using test dataset: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure your test dataset is correctly placed.")