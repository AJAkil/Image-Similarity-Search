import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from encoder_models import *
from decoder_models import *
import CONFIG
import argparse


class FolderDataset(Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args:
    main_dir : directory where images are stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image


def create_embedding(encoder, full_loader, embedding_dim, device):
    """
    Creates embedding using encoder from dataloader.
    encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
    full_loader: PyTorch dataloader, containing (images, images) over entire dataset.
    embedding_dim: Tuple (c, h, w) Dimension of embedding = output of encoder dimensions.
    device: "cuda" or "cpu"
    Returns: Embedding of size (num_images_in_loader + 1, c, h, w)
    """
    # Set encoder to eval mode.
    encoder.eval()
    # Just a place holder for our 0th image embedding.
    embedding = torch.randn(embedding_dim)

    # Again we do not compute loss here so. No gradients.
    with torch.no_grad():
        for batch_idx, image in enumerate(full_loader):

            # Move images to device
            img = image['image_X'].to(device)

            # Get encoder outputs and move outputs to cpu
            enc_output = encoder(img)
            enc_output = enc_output.cpu()
            # Keep adding these outputs to embeddings.
            embedding = torch.cat((embedding, enc_output), 0)

    # Return the embeddings
    print(f'The Embedding Shape: {embedding.size()}')
    return embedding


def create_embedding_full(encoder, full_loader, embedding_dim, device, is_vae=False):
    encoder.eval()
    embedding = torch.randn(embedding_dim)
    # print(embedding.shape)

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(full_loader):
            train_img = train_img.to(device)

            if is_vae:
                enc_output, _, _ = encoder(train_img)
            else:
                enc_output = encoder(train_img)
            # print(enc_output.shape)
            enc_output = enc_output.cpu()
            embedding = torch.cat((embedding, enc_output), 0)
            # print(embedding.shape)

    print(f'The Embedding Shape: {embedding.size()}')
    return embedding


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(
        description='flags')

    # Add the arguments
    parser.add_argument('--arch_type',
                        metavar='arch_type',
                        type=str,
                        help='the type of neural network')

    parser.add_argument('--model_name',
                        metavar='model_name',
                        type=str,
                        help='the name of the model')

    parser.add_argument('--dataset',
                        metavar='dataset',
                        type=str,
                        help='dataset to work eith')

    parser.add_argument('--dataset_type',
                        metavar='dataset type',
                        type=str,
                        help='dataset type to work eith')

    # Execute the parse_args() method
    args = parser.parse_args()
    arch_type = args.arch_type
    model_name = args.model_name
    dataset = args.dataset
    dataset_type = args.dataset_type

    if arch_type == 'vae':
        is_vae = True

        if model_name == 'res50':

            print('choosing resnet 50 vae as encoder')
            encoder = Resnet50VaeEncoder()

        if model_name == 'res101':

            print('choosing resnet 101 vae as encoder')
            encoder = Resnet101VaeEncoder()

        if model_name == 'res18':

            print('choosing resnet 18 vae as encoder')
            encoder = Resnet18FullVaeEncoder()

        if model_name == 'effb0':

            print('choosing eff net b0 vae as encoder')
            encoder = EfficientNetB0VaeEncoder()

    else:
        is_vae = False
        if model_name == 'res101':

            # better one than efficient net b0
            print('choosing resnet 101 as encoder')
            encoder = ResnetEncoder()

        if model_name == 'effb0':

            print('choosing effb0 as encoder')
            encoder = EfficientNetB0Encoder()

    encoder_path = CONFIG.ANIMAL_10_ENCODER_PATH if dataset == 'animal-10' else CONFIG.CALTECH_ENCODER_PATH
    embedding_path = CONFIG.ANIMAL_10_EMBEDDING_PATH if dataset == 'animal-10' else CONFIG.CALTECH_EMBEDDING_PATH

    if dataset_type == 'test':
        data_path = CONFIG.ANIMAL_10_TEST_DATA_PATH if dataset == 'animal-10' else CONFIG.CALTECH256_TEST_DATA_PATH
    elif dataset_type == 'full':
        data_path = CONFIG.ANIMAL_10_ALL_DATA_PATH if dataset == 'animal-10' else CONFIG.CALTECH256_ALL_DATA_PATH

    transforms = T.Compose([T.Resize([int(224), int(224)]), T.ToTensor()])

    print("********** Creating Dataset **********")
    print(f'Creating Dataset for: {data_path}')
    processed_dataset = FolderDataset(data_path, transforms)
    print("********** Dataset Created **********")

    print("********** Creating DataLoader **********")
    data_loader = torch.utils.data.DataLoader(
        processed_dataset, batch_size=32
    )

    print(f'Length of full dataset: {len(processed_dataset)}')

    print("---- Creating Embeddings for the dataset ---- ")

    print(f'Given Encoder Path: {encoder_path}')

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the state dict of encoder
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)

    embedding = create_embedding_full(
        encoder, data_loader, (1, CONFIG.EMBEDDING_DIMENSION), device, is_vae
    )

    # Convert embedding to numpy and save them
    numpy_embedding = embedding.cpu().detach().numpy()
    num_images = numpy_embedding.shape[0]

    # Dump the embeddings for complete dataset, not just train
    flattened_embedding = numpy_embedding.reshape((num_images, -1))

    print(f'Creating Embedding At: {embedding_path}')
    np.save(embedding_path, flattened_embedding)
