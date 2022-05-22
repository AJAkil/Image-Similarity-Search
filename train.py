import time
import torch.nn as nn
import torch
import wandb
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import numpy as np

from encoder_models import *
from decoder_models import *
from utils import LRScheduler, EarlyStopping
from training_utils import *
from custom_Image_dataset import ImageDataset
import CONFIG
from collections import Counter


wandb.init(project="image-similarity-search", entity="ajwadakil")


def set_up_device():
    print(torch.__version__)
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current Device: {device}')

    return device


def setup_dataset():
    # get the dataset ready
    print(f'Working with {CONFIG.ANIMAL_DATA_PATH_CSV} right now: ')
    df = pd.read_csv(CONFIG.ANIMAL_DATA_PATH_CSV)

    X = df.image_path.values  # image paths
    y = df.target.values  # targets

    batch_size = 32

    (xtrain, x_val_test, ytrain, y_val_test) = train_test_split(X, y, stratify=y,
                                                                test_size=0.10, random_state=79)

    (x_val, x_test, y_val, y_test) = train_test_split(x_val_test, y_val_test, stratify=y_val_test,
                                                      test_size=0.5, random_state=81)

    df = pd.DataFrame(list(zip(x_test, y_test)),
                      columns=['image_path', 'class'])

    df.to_csv('./test_data_animal_10.csv')

    print(f"Training instances: {len(xtrain)}")
    print(f"Validation instances: {len(x_val)}")
    print(f"Testing instances: {len(x_test)}")

    train_data = ImageDataset(xtrain, ytrain, tfms=1)
    validation_data = ImageDataset(x_val, y_val, tfms=0)
    test_data = ImageDataset(x_test, y_test, tfms=0)
    full_dataset = ImageDataset(X, y, tfms=0)

    # dataloaders
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    full_dataset_data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, valid_data_loader, test_data_loader, full_dataset_data_loader

def setup_dataset_with_oversampling():

    print('Oversampling')
    # get the dataset ready
    print(f'Working with {CONFIG.ANIMAL_DATA_PATH_CSV} right now: ')
    df = pd.read_csv(CONFIG.ANIMAL_DATA_PATH_CSV)

    X = df.image_path.values  # image paths
    y = df.target.values  # targets

    batch_size = 32

    (xtrain, x_val_test, ytrain, y_val_test) = train_test_split(X, y, stratify=y,
                                                                test_size=0.10, random_state=79)

    (x_val, x_test, y_val, y_test) = train_test_split(x_val_test, y_val_test, stratify=y_val_test,
                                                      test_size=0.5, random_state=81)

    df = pd.DataFrame(list(zip(x_test, y_test)),
                      columns=['image_path', 'class'])

    df.to_csv('./test_data_animal_10.csv')

    print(f"Training instances: {len(xtrain)}")
    print(f"Validation instances: {len(x_val)}")
    print(f"Testing instances: {len(x_test)}")


    # perform weighted sampling of the dataset
    count = Counter(ytrain)
    # print(ytrain)

    class_count = np.array([c for _, c in count.items()])

    weight = 1. / class_count

    print(len(ytrain))
    print(len(weight))

    samples_weight = np.array([weight[int(t)] for t in ytrain])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_data = ImageDataset(xtrain, ytrain, tfms=1)
    validation_data = ImageDataset(x_val, y_val, tfms=0)
    test_data = ImageDataset(x_test, y_test, tfms=0)

    # dataloaders
    train_data_loader = DataLoader(train_data, batch_size=batch_size,sampler=sampler)
    valid_data_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_data_loader, valid_data_loader, test_data_loader


def train_models(device, learning_rate=0.001, epochs=100):
    train_data_loader, valid_data_loader, test_data_loader = setup_dataset_with_oversampling()

    criterion = nn.MSELoss()  # We use Mean squared loss which computes difference between two images.

    encoder = ResnetEncoder()  # encoder model
    decoder = SimpleConvolutionDecoderResnet()  # decoder model

    # Load the state dict of encoder
    # encoder.load_state_dict(torch.load('./encoder_model_.pt', map_location=device))
    # encoder.to(device)

    encoder.to(device)
    decoder.to(device)

    encoder_params = list(encoder.parameters())
    decoder_params = list(decoder.parameters())
    autoencoder_params = encoder_params + decoder_params
    optimizer = torch.optim.Adam(autoencoder_params, lr=learning_rate)  # Adam Optimizer
    lr_scheduler = LRScheduler(optimizer=optimizer)
    earlyStopping = EarlyStopping()

    max_loss = 1e10

    start = time.time()

    for epoch in tqdm(range(epochs)):

        train_loss = train_step(encoder, decoder, train_data_loader, criterion, optimizer, device=device)

        print(f"Epochs = {epoch}, Training Loss : {train_loss}")
        wandb.log({"\ntrain_loss": train_loss, "epoch": epoch})

        val_loss = val_step(encoder, decoder, valid_data_loader, criterion, device=device)

        print(f"Epochs = {epoch}, Validation Loss : {val_loss}")
        wandb.log({"\nvalid_loss": val_loss, "epoch": epoch})

        # Simple Best Model saving
        if val_loss < max_loss:
            print("\nValidation Loss decreased, saving new best model")
            print("\nValidation Loss decreased")
            torch.save(encoder.state_dict(), "checkpoints/animal-10/ae/encoder_resnet101_os.pt")
            torch.save(decoder.state_dict(), "checkpoints/animal-10/ae/decoder_resnet101_os.pt")
            max_loss = val_loss

        # updating learning rate with validation loss
        lr_scheduler(val_loss)

        # earcly stopping
        earlyStopping(val_loss)
        if earlyStopping.early_stop:
            break

    end = time.time()
    print(f"Training time: {(end - start) / 60:.3f} minutes")


def train_vae(device, learning_rate=0.001, epochs=100):
    train_data_loader, valid_data_loader, test_data_loader, _ = setup_dataset()

    criterion = F.mse_loss  # We use Mean squared loss which computes difference between two images.

    encoder = Resnet18FullVaeEncoder()  # encoder model
    decoder = SimpleConvolutionDecoderResnetVAE()  # decoder model

    # Load the state dict of encoder
    # encoder.load_state_dict(torch.load('./encoder_model_.pt', map_location=device))
    # encoder.to(device)

    encoder.to(device)
    decoder.to(device)

    encoder_params = list(encoder.parameters())
    decoder_params = list(decoder.parameters())
    autoencoder_params = encoder_params + decoder_params
    optimizer = torch.optim.Adam(autoencoder_params, lr=learning_rate)  # Adam Optimizer
    lr_scheduler = LRScheduler(optimizer=optimizer, patience=4)
    earlyStopping = EarlyStopping(patience_epoch=8)

    max_loss = 1e10

    start = time.time()

    for epoch in tqdm(range(epochs)):

        train_loss, train_kl_loss = train_step_vae(encoder, decoder, train_data_loader, criterion, optimizer, device=device)

        print("*"*50)

        print(f"Epochs = {epoch}, Training Loss : {train_loss} KL Loss: {train_kl_loss}")
        
        #wandb.log({"\ntrain_loss": train_loss, "epoch": epoch})
        #wandb.log({"\ntrain_kl_loss": train_kl_loss, "epoch": epoch})


        val_loss, val_kl_loss = val_step_vae(encoder, decoder, valid_data_loader, criterion, device=device)

        print(f"Epochs = {epoch}, Validation Loss : {val_loss} KL Loss: {val_kl_loss}")

        #wandb.log({"\nvalid_loss": val_loss, "epoch": epoch})
        #wandb.log({"\nvalid_kl_loss": val_kl_loss, "epoch": epoch})

        # Simple Best Model saving
        if val_loss < max_loss:
            print("\nValidation Loss decreased, saving new best model")
            #print("\nValidation Loss decreased")
            torch.save(encoder.state_dict(), "checkpoints/animal-10/vae/resnet18/encoder_resnet18_full_run2.pt")
            torch.save(decoder.state_dict(), "checkpoints/animal-10/vae/resnet18/decoder_resnet18_full_run2.pt")
            max_loss = val_loss

        print("*"*50)

        # updating learning rate with validation loss
        lr_scheduler(val_loss)

        # earcly stopping
        earlyStopping(val_loss)
        if earlyStopping.early_stop:
            break

    end = time.time()
    print(f"Training time: {(end - start) / 60:.3f} minutes")



def train_vae_cosine_annealing(device, learning_rate=0.02, epochs=100):
    train_data_loader, valid_data_loader, test_data_loader = setup_dataset_with_oversampling()

    criterion = F.mse_loss  # We use Mean squared loss which computes difference between two images.

    encoder = Resnet50VaeEncoder()  # encoder model
    decoder = SimpleConvolutionDecoderResnetVAE()  # decoder model

    # Load the state dict of encoder
    # encoder.load_state_dict(torch.load('./encoder_model_.pt', map_location=device))
    # encoder.to(device)

    encoder.to(device)
    decoder.to(device)
    steps = len(train_data_loader)
    print(steps)

    encoder_params = list(encoder.parameters())
    decoder_params = list(decoder.parameters())
    autoencoder_params = encoder_params + decoder_params
    optimizer = torch.optim.Adam(autoencoder_params, lr=learning_rate)  # Optimizer
    earlyStopping = EarlyStopping(patience_epoch=8)

    warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)
    mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

    max_loss = 1e10

    start = time.time()

    for epoch in tqdm(range(epochs)):

        encoder.train()
        decoder.train()

        for batch_idx, image in enumerate(train_data_loader):
            # Move images to device
            train_img = image['image_X'].to(device)
            target_img = image['image_Y'].to(device)

            # Feed the train images to encoder
            encoder_output, z_mean, z_logvar = encoder(train_img)
            decoder_output = decoder(encoder_output)

            kl_div, batch_size = calculate_kl_divergence(z_logvar=z_logvar, z_mean=z_mean)

            # asserting the batch size shape here
            assert batch_size == decoder_output.size(0)

            # Decoder output is reconstructed image
            loss, kl_loss = get_hybrid_loss(criterion, decoder_output, target_img, gamma=0.7, kl_div_loss=kl_div)
            # print(loss)
            # print(kl_loss)

            # zero grad the optimizer
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        train_loss, train_kl_loss = loss.item(), kl_loss.item()

        print("*"*50)

        print(f"Epochs = {epoch}, Training Loss : {train_loss} KL Loss: {train_kl_loss}")
        
        wandb.log({"\ntrain_loss": train_loss, "epoch": epoch})
        wandb.log({"\ntrain_kl_loss": train_kl_loss, "epoch": epoch})


        wandb.log({"\nWarmup learning rate": warmupscheduler.get_lr()[0], "epoch": epoch})
        wandb.log({"\nCosine Annealing learning rate": mainscheduler.get_lr()[0], "epoch": epoch})

        if epoch < 10:
            warmupscheduler.step()
        if epoch >= 10:
            mainscheduler.step()


        val_loss, val_kl_loss = val_step_vae(encoder, decoder, valid_data_loader, criterion, device=device)

        print(f"Epochs = {epoch}, Validation Loss : {val_loss} KL Loss: {val_kl_loss}")

        wandb.log({"\nvalid_loss": val_loss, "epoch": epoch})
        wandb.log({"\nvalid_kl_loss": val_kl_loss, "epoch": epoch})

        # Simple Best Model saving
        if val_loss < max_loss:
            print("\nValidation Loss decreased, saving new best model")
            #print("\nValidation Loss decreased")
            torch.save(encoder.state_dict(), "checkpoints/animal-10/vae/resnet50/encoder_resnet50_run3_os_gamma.pt")
            torch.save(decoder.state_dict(), "checkpoints/animal-10/vae/resnet50/decoder_resnet50_run3_os_gamma.pt")
            max_loss = val_loss

        print("*"*50)
        

        # earcly stopping
        earlyStopping(val_loss)
        if earlyStopping.early_stop:
            break

    end = time.time()
    print(f"Training time: {(end - start) / 60:.3f} minutes")


# def train_vae_cosine_annealing(device, learning_rate=0.02, epochs=100):
#     train_data_loader, valid_data_loader, test_data_loader = setup_dataset_with_oversampling()

#     criterion = F.mse_loss  # We use Mean squared loss which computes difference between two images.

#     encoder = Resnet50VaeEncoder()  # encoder model
#     decoder = SimpleConvolutionDecoderResnetVAE()  # decoder model

#     # Load the state dict of encoder
#     # encoder.load_state_dict(torch.load('./encoder_model_.pt', map_location=device))
#     # encoder.to(device)

#     encoder.to(device)
#     decoder.to(device)
#     steps = len(train_data_loader)
#     print(steps)

#     encoder_params = list(encoder.parameters())
#     decoder_params = list(decoder.parameters())
#     autoencoder_params = encoder_params + decoder_params
#     optimizer = torch.optim.Adam(autoencoder_params, lr=learning_rate)  # Optimizer
#     earlyStopping = EarlyStopping(patience_epoch=8)

#     warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)
#     mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

#     max_loss = 1e10

#     start = time.time()

#     for epoch in tqdm(range(epochs)):

#         encoder.train()
#         decoder.train()

#         for batch_idx, image in enumerate(train_data_loader):
#             # Move images to device
#             train_img = image['image_X'].to(device)
#             target_img = image['image_Y'].to(device)

#             # Feed the train images to encoder
#             encoder_output, z_mean, z_log_var = encoder(train_img)
#             decoder_output = decoder(encoder_output)

            
#             kl_div = -0.5 * torch.sum(1 + z_log_var 
#                                       - z_mean**2 
#                                       - torch.exp(z_log_var), 
#                                       axis=1) # sum over latent dimension

#             batchsize = kl_div.size(0)
#             kl_div = kl_div.mean() # average over batch dimension
    
#             pixelwise = criterion(decoder_output, target_img, reduction='none')
#             pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
#             pixelwise = pixelwise.mean() # average over batch dimension
            
#             loss = 1*pixelwise + kl_div

#             # print(loss)
#             # print(kl_loss)

#             # zero grad the optimizer
#             optimizer.zero_grad()

#             loss.backward()

#             optimizer.step()

#         train_loss, train_kl_loss = loss.item(), kl_div.item()

#         print("*"*50)

#         print(f"Epochs = {epoch}, Training Loss : {train_loss} KL Loss: {train_kl_loss}")
        
#         wandb.log({"\ntrain_loss": train_loss, "epoch": epoch})
#         wandb.log({"\ntrain_kl_loss": train_kl_loss, "epoch": epoch})


#         wandb.log({"\nWarmup learning rate": warmupscheduler.get_lr()[0], "epoch": epoch})
#         wandb.log({"\nCosine Annealing learning rate": mainscheduler.get_lr()[0], "epoch": epoch})

#         if epoch < 14:
#             warmupscheduler.step()
#         if epoch >= 14:
#             mainscheduler.step()


#         val_loss, val_kl_loss = val_step_vae(encoder, decoder, valid_data_loader, criterion, device=device)

#         print(f"Epochs = {epoch}, Validation Loss : {val_loss} KL Loss: {val_kl_loss}")

#         wandb.log({"\nvalid_loss": val_loss, "epoch": epoch})
#         wandb.log({"\nvalid_kl_loss": val_kl_loss, "epoch": epoch})

#         # Simple Best Model saving
#         if val_loss < max_loss:
#             print("\nValidation Loss decreased, saving new best model")
#             #print("\nValidation Loss decreased")
#             torch.save(encoder.state_dict(), "checkpoints/animal-10/vae/resnet50/encoder_resnet50_run4_os.pt")
#             torch.save(decoder.state_dict(), "checkpoints/animal-10/vae/resnet50/decoder_resnet50_run4_os.pt")
#             max_loss = val_loss

#         print("*"*50)
        

#         # earcly stopping
#         earlyStopping(val_loss)
#         if earlyStopping.early_stop:
#             break

#     end = time.time()
#     print(f"Training time: {(end - start) / 60:.3f} minutes")

if __name__ == "__main__":
    device = set_up_device()
    #train_models(device, learning_rate=0.0016)

    wandb.config = {
    "learning_rate": 0.0007,
    "epochs": 100,
    "batch_size": 32
    }

    train_vae_cosine_annealing(device, learning_rate=0.0014)
