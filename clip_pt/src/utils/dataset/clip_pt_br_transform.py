import random
import torch
import numpy as np
import torchvision.transforms.functional as F

class Clipt_PT_Transform(torch.nn.Module):
    def __init__(self,augmentation_rate,num_transfor):
        super().__init__()
        self.augmentation_rate=augmentation_rate
        self.num_transfor=num_transfor

    def _apply_op(self, img,op_name):
        if op_name == "Posterize":
            img = F.posterize(img, random.randint(6,8))
        elif op_name == "Rotate":        
            img = F.rotate(img, random.choice([random.randint(5,15),random.randint(340,355)]))
        elif op_name == "Equalize":
            img = F.equalize(img)
        elif op_name == "Gaussian_Blur":
            img = F.gaussian_blur(img, random.randrange(3,9,2))
        else:
            magnitude = random.randint(5,15)/10
            if op_name == "Saturation_Factor":
                img = F.adjust_saturation(img, magnitude)
            elif op_name == "Sharpness":
                img = F.adjust_sharpness(img, magnitude)
            elif op_name == "Brightness":
                img = F.adjust_brightness(img, magnitude)
            elif op_name == "Contrast":
                img = F.adjust_contrast(img, magnitude)
            elif op_name == "Gamma":
                img = F.adjust_gamma(img,magnitude)
            else:
                raise ValueError(f"The provided operator {op_name} is not recognized.")
        return img

    def _select_transform(self):
        transformations = np.array(['Posterize','Rotate','Equalize','Saturation_Factor','Sharpness','Brightness','Contrast','Gamma','Gaussian_Blur'])
        return transformations[random.sample(range(0, len(transformations)), self.num_transfor)]

    def forward(self, image):
        transformantions_selected = self._select_transform()
        if random.random() > self.augmentation_rate:
            for transform in transformantions_selected:
                image = self._apply_op(image,transform)
        return image
