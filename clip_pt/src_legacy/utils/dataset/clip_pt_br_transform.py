import random

import torch
import torchvision.transforms.functional as F


class CliptPTTransform(torch.nn.Module):
    def __init__(self, n_transformations=1, augmentation_rate=0.8):
        """

        :param n_transformations: number of simultaneous transformations
        :param augmentation_rate: percentage of the dataset that will be transformed
        """
        super().__init__()
        assert 0.0 <= augmentation_rate <= 1.0
        assert n_transformations >= 1

        self.augmentation_rate = augmentation_rate
        self.n_transformations = n_transformations

    def __apply_op(self, img, op_name):
        if op_name == "Posterize":
            img = F.posterize(img, random.randint(6, 8))
        elif op_name == "Rotate":
            img = F.rotate(img, random.choice([random.randint(5, 15),
                                               random.randint(340, 355)]))
        elif op_name == "Equalize":
            img = F.equalize(img)
        elif op_name == "Gaussian_Blur":
            kernel_size = random.randrange(3, 9, 2)
            img = F.gaussian_blur(img, kernel_size=[kernel_size, kernel_size])
        else:
            magnitude = random.randint(5, 15) / 10
            if op_name == "Saturation_Factor":
                img = F.adjust_saturation(img, magnitude)
            elif op_name == "Sharpness":
                img = F.adjust_sharpness(img, magnitude)
            elif op_name == "Brightness":
                img = F.adjust_brightness(img, magnitude)
            elif op_name == "Contrast":
                img = F.adjust_contrast(img, magnitude)
            elif op_name == "Gamma":
                img = F.adjust_gamma(img, magnitude)
            else:
                raise ValueError(f"The provided operator {op_name} is not recognized.")
        return img

    def __select_transform(self):
        transformations = ['Posterize', 'Rotate', 'Equalize', 'Saturation_Factor', 'Sharpness',
                           'Brightness', 'Contrast', 'Gamma', 'Gaussian_Blur']
        return random.sample(transformations, k=self.n_transformations)

    def forward(self, image):
        transformations_selected = self.__select_transform()
        if random.random() <= self.augmentation_rate:
            for transform in transformations_selected:
                image = self.__apply_op(image, transform)

        return image
