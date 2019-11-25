import numpy as np
import torch
import random
from torchvision import transforms
from scipy.ndimage import gaussian_filter


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']

        img = img.astype(np.float32)

        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))

        mask = mask.astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = gaussian_filter(img, sigma=random.random())

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):

    def __call__(self, sample):

        img = sample['image']
        mask = sample['label']

        if random.random() < 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.shape[1], img.shape[0]

        if w > h:
            oh = self.crop_size[0]
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size[1]
            oh = int(1.0 * h * ow / w)

        img = np.array(Image.fromarray(img).resize(ow, oh)) 
        mask = np.array(Image.fromarray(mask).resize(ow, oh), PIL.Image.NEAREST)

        # center crop
        w, h = img.shape[1], img.shape[0]
        x1 = int(round((w - self.crop_size[1]) / 2.))
        y1 = int(round((h - self.crop_size[0]) / 2.))
        img = img[y1: y1 + self.crop_size[0], x1: x1 + self.crop_size[1], :]
        mask = mask[y1: y1 + self.crop_size[0], x1: x1 + self.crop_size[1]]

        return {'image': img,
                'label': mask}


def transform_training_sample(image, target, base_size):
    composed_transforms = transforms.Compose([
        RandomHorizontalFlip(),
        RandomGaussianBlur(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor()
    ])

    return composed_transforms({'image': image, 'label': target})


def transform_validation_sample(image, target, base_size=None):
    composed_transforms = transforms.Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor()
    ])

    return composed_transforms({'image': image, 'label': target})
