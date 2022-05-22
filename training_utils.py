import torch
import torch.nn.functional as F


def calculate_kl_divergence(z_logvar, z_mean):
    arg = 1 + z_logvar - z_mean ** 2 - torch.exp(z_logvar)
    kl_div = -0.5 * torch.sum(arg, axis=1)  # sum over latent dimension

    batchsize = kl_div.size(0)
    kl_div = kl_div.mean()  # average over batch dimension

    return kl_div, batchsize


def get_hybrid_loss(loss_function, decoder_output, target_img, gamma, kl_div_loss):
    # print(target_img)
    # print('decoder output: ', decoder_output)
    pixel_loss = loss_function(decoder_output, target_img, reduction='none')

    test = pixel_loss.view(decoder_output.size(0), -1)
    # print(decoder_output.size())
    # print(test.size())

    pixel_loss = pixel_loss.view(decoder_output.size(0), -1).sum(axis=1)  # sum over pixels
    # print('In loss calculation:', pixel_loss.size())
    pixel_loss = pixel_loss.mean()  # average over batch dimension

    # print('pixel loss:', pixel_loss.item())
    # print('kl loss: ', kl_div_loss.item())

    # pixel_loss = F.mse_loss(decoder_output, target_img, reduction='sum')
    # print('with functional api: ', pixel_loss)

    hybrid_loss = gamma * pixel_loss + kl_div_loss

    return hybrid_loss, kl_div_loss


def train_step_vae(encoder, decoder, train_data_loader, loss_function, optimizer, device):
    #  Set networks to train mode.

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
        loss, kl_loss = get_hybrid_loss(loss_function, decoder_output, target_img, 1, kl_div)

        # zero grad the optimizer
        optimizer.zero_grad()

        loss.backward()

        # update the model parameters
        optimizer.step()

    # Return the loss
    return loss.item(), kl_loss.item()


def train_step(encoder, decoder, train_data_loader, loss_function, optimizer, device):
    """
    """
    #  Set networks to train mode.

    encoder.train()
    decoder.train()

    for batch_idx, image in enumerate(train_data_loader):
        # Move images to device
        train_img = image['image_X'].to(device)
        target_img = image['image_Y'].to(device)

        # Zero grad the optimizer
        optimizer.zero_grad()

        # Feed the train images to encoder
        encoder_output = encoder(train_img)
        decoder_output = decoder(encoder_output)

        # Decoder output is reconstructed image
        loss = loss_function(decoder_output, target_img)
        loss.backward()
        optimizer.step()

    # Return the loss
    return loss.item()


def val_step(encoder, decoder, valid_data_loader, loss_function, device):
    """
    """

    # Set to eval mode.
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch_idx, img in enumerate(valid_data_loader):
            # Move to device
            train_img = img['image_X'].to(device)
            target_img = img['image_Y'].to(device)

            # Feed the train images to encoder
            encoder_output = encoder(train_img)
            decoder_output = decoder(encoder_output)

            # Decoder output is reconstructed image

            # Validation loss for encoder and decoder.
            loss = loss_function(decoder_output, target_img)
    # Return the loss
    return loss.item()


# def val_step_vae(encoder, decoder, valid_data_loader, loss_function, device):
#     """
#     """

#     # Set to eval mode.
#     encoder.eval()
#     decoder.eval()
#     encoder.is_training = False

#     with torch.no_grad():
#         for batch_idx, img in enumerate(valid_data_loader):
#             # Move to device
#             train_img = img['image_X'].to(device)
#             target_img = img['image_Y'].to(device)

#             # Feed the train images to encoder
#             encoder_output, z_mean, z_logvar = encoder(train_img)
#             decoder_output = decoder(encoder_output)

#             kl_div, batch_size = calculate_kl_divergence(z_logvar=z_logvar, z_mean=z_mean)

#             # asserting the batch size shape here
#             assert batch_size == decoder_output.size(0)

#             loss, kl_loss = get_hybrid_loss(loss_function, decoder_output, target_img, 1, kl_div)

#     # Return the loss
#     encoder.is_training = True
#     return loss.item(), kl_loss.item()


def val_step_vae(encoder, decoder, valid_data_loader, loss_function, device):
    """
    """

    # Set to eval mode.
    encoder.eval()
    decoder.eval()
    encoder.is_training = False

    with torch.no_grad():
        for batch_idx, img in enumerate(valid_data_loader):
            # Move to device
            train_img = img['image_X'].to(device)
            target_img = img['image_Y'].to(device)

            # Feed the train images to encoder
            encoder_output, z_mean, z_log_var = encoder(train_img)
            decoder_output = decoder(encoder_output)

            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      axis=1) # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension
    
            pixelwise = loss_function(decoder_output, target_img, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension
            
            loss = 1*pixelwise + kl_div

    # Return the loss
    encoder.is_training = True
    return loss.item(), kl_div.item()
