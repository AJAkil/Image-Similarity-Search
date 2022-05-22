# custom dataset

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageDataset(Dataset):
    def __init__(self, images, labels=None, tfms=None):
        self.X = images
        self.y = labels

        self.animal_10_data_mean = [0.5181, 0.5009, 0.4129]
        self.animal_10_data_std = [0.2657, 0.2607, 0.2786]

        self.img_net_mean = [0.485, 0.456, 0.406]
        self.img_net_std = [0.229, 0.224, 0.225]

        self.caltech256_mean = [0.1976, 0.2117, 0.2988]
        self.caltech256_std = [1.3567, 1.3633, 1.4053]

        # apply augmentations
        if tfms == 0:  # if validating
            self.aug = T.Compose([T.Resize(255),
                                  T.CenterCrop(224),
                                  T.ToTensor(),
                                  #T.Normalize(mean=self.animal_10_data_mean, std=self.animal_10_data_std)
                                  ])

            # self.aug  = T.Compose([T.Resize([int(224), int(224)]), 
            #             T.ToTensor(), 
            #             #T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
            #             ])
        else:  # if training
            self.aug = T.Compose([T.RandomRotation(25),
                                  T.RandomResizedCrop(224),
                                  T.RandomHorizontalFlip(),
                                  T.ToTensor(),
                                  #T.Normalize(mean=self.animal_10_data_mean, std=self.animal_10_data_std)
                                  ])
            # self.aug = T.Compose([T.Resize([int(224), int(224)]), 
            #             T.ToTensor(), 
            #             #T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
            #             ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):

        image = Image.open(self.X[i])
        image = image.convert('RGB')
        image = self.aug(image)
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            'image_X': image,
            'image_Y': image
        }
