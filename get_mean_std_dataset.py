import torch
import pandas as pd
from custom_Image_dataset import ImageDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
torch.manual_seed(0)
import random
random.seed(0)

# get the dataset ready
df = pd.read_csv('data.csv')

X = df.image_path.values # image paths
y = df.target.values # targets

batch_size = 32

(xtrain, x_val_test, ytrain, y_val_test) = train_test_split(X, y, stratify=y,
	test_size=0.10, random_state=79)

    
print(f"Training instances: {len(xtrain)}")


train_data = ImageDataset(xtrain, ytrain, tfms=1)

# dataloaders
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data_index, data in enumerate(loader):
        X, Y = data['image_X'].to(device), data['image_Y'].to(device)
        channels_sum += torch.mean(X, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(X ** 2, dim=[0, 2, 3])
        num_batches += 1

    print('num of batches:', num_batches)

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


train_mean, train_std = get_mean_std(train_data_loader)

print(train_mean)
print(train_std)
