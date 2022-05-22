from collections import defaultdict
import numpy as np
import torch
import os
from PIL import Image
import torchvision.transforms as T
from encoder_models import *
from decoder_models import *
import CONFIG
import albumentations
import argparse

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import albumentations
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import csv

def preprocess_query_images(image_path):
    """
    Preprocess an Image and convert it to an image Tensor
    :param image:
    :return:
    """

    aug = albumentations.Compose([
        albumentations.Resize(224, 224, always_apply=True),
        albumentations.Normalize()
    ])

    print(image_path)

    image = Image.open(image_path).convert("RGB")
    image = aug(image=np.array(image))['image']
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image_tensor = torch.tensor(image, dtype=torch.float)
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    print('image tensor size:', image_tensor.size())

    return image_tensor





def load_image_tensor(image_path, device):
    """
    Loads a given image to device.
    Args:
    image_path: path to image to be loaded.
    device: "cuda" or "cpu"
    """
    transforms = T.Compose([T.Resize([int(224), int(224)]), T.ToTensor()])
    image_tensor = transforms(Image.open(image_path))
    #image_tensor = T.ToTensor()(Image.open(image_path))
    image_tensor = image_tensor.unsqueeze(0)
    #print(image_tensor.shape)
    # input_images = image_tensor.to(device)
    return image_tensor


def compute_similar_images(image_path, num_images, embeddings_all_image, device, is_vae=False):
    """
    Given an image and number of similar images to search.
    Returns the num_images closest neares images.
    Args:
    image: Image whose similar images are to be found.
    num_images: Number of similar images to find.
    embeddings_all_image : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    device : "cuda" or "cpu" device.
    """
    image_tensor = load_image_tensor(image_path, device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        #print(type(image_tensor))
        #print(image_tensor.size())

        encoder.is_training = False

        if is_vae:
            image_embedding, z_mean, z_logvar = encoder(image_tensor)
        else:
            image_embedding = encoder(image_tensor)
        
        image_embedding = image_embedding.cpu().detach().numpy()

    #print(f'Embedding Computation Done!, The Embedding Size: {image_embedding.shape}')

    embedding_flattened = image_embedding.reshape((image_embedding.shape[0], -1))
    #
    #print(f'Embedding Flattening Done!. Shape is: {embedding_flattened.shape}')

    #print('Computing Nearest Neighbors!')
    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embeddings_all_image)


    #print('Computation Done! Finding Nearest Neighbors!')
    _, indices = knn.kneighbors(image_embedding)
    indices_list = indices.tolist()

    #print(indices_list)
    return indices_list


def plot_similar_images(indices_list):
    """
    Plots images that are similar to indices obtained from computing simliar images.
    Args:
    indices_list : List of List of indexes. E.g. [[1, 2, 3]]
    """
    
    df = pd.read_csv('./data.csv')

    print(indices_list)
    indices = indices_list[0]
    for index in indices:
        if index == 0:
            # index 0 is a dummy embedding.
            pass
        else:
            #img_name = str(index - 1) + ".jpg"
            img_path = df.iloc[index-1, 0]
            print(img_path)
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.show()
            #img.save(f"../outputs/query_image_3/recommended_{index - 1}.jpg")


def plot_similar_images_(indices_list, data_path):
    """
    Plots images that are similar to indices obtained from computing simliar images.
    Args:
    indices_list : List of List of indexes. E.g. [[1, 2, 3]]
    """
    dir = data_path
    image_lists = os.listdir(dir)
    #print('in plotting f()', image_lists)

    indices = indices_list[0]
    print(indices)
    for index in indices:
        if index == 0:
            # index 0 is a dummy embedding.
            pass
        else:
            img_name = image_lists[index - 1]
            #print(f'Image Displayed: {img_name}')
            img_path = os.path.join(dir+ img_name)
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.show()
            #img.save(f"../outputs/query_image_3/recommended_{index - 1}.jpg")


def compute_metrics( embedding, device, num_classes=10, num_images=5, is_vae=False):

    confusion_matrix = np.zeros((num_classes,num_classes))

    class_index_to_test_image_path = defaultdict(list)
    text_image_path_to_class_index = {}
    

    with open('test_data_animal_10.csv') as csv_file:
      
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                line_count += 1
                #print(f'from csv: {row[1].split("/")[-1]}')
                class_index_to_test_image_path[int(float(row[2]))].append(row[1].split("/")[-1])
                text_image_path_to_class_index[row[1].split("/")[-1]] = int(float(row[2]))

    #print(text_image_path_to_class_index)

    dir_list = os.listdir("test-data-animals-10")
    

    for class_index in range(num_classes):

        example_index = class_index_to_test_image_path[class_index][:20]

        for y_test_index in example_index:
            
            path = os.path.join("test-data-animals-10", y_test_index)
            indices = compute_similar_images(path, num_images, embedding, device, is_vae)

            for image_index in indices:
                
                for each_image_index in image_index:
                    #print(f'Trying : {each_image_index - 1}')
                    image_path = dir_list[each_image_index - 1]
                    #print('image path: ', image_path)
                    image_class_index = text_image_path_to_class_index[image_path]
                    confusion_matrix[class_index,image_class_index] += 1

        
    print(confusion_matrix)

    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) 
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)       

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    #print("Recall : " + str(TPR)) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)    
    #print("Precision : " + str(PPV)) 
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN) 
    #print("Accuracy : " + str(ACC)) 
    #print(type(TPR))
    #print(TPR.shape)
    #print(type(PPV))
    #print(PPV.shape)

    TPR = np.reshape(TPR,(num_classes,1))
    PPV = np.reshape(PPV,(num_classes,1))
    

    F1 = (2*TPR*PPV)/(TPR+PPV)
    #print("F1 Score : " + str(F1)) 
    
    print(f'Macro F1: {np.mean(F1)}')
    print(f'Macro Precision: {np.mean(PPV)}')
    print(f'Macro Recall: {np.mean(TPR)}')


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


    parser.add_argument('--K',
                        metavar='num_neighbor',
                        type=int,
                        help='the number of neighbors to return')

    parser.add_argument('--dataset',
                        metavar='dataset',
                        type=str,
                        help='dataset to work eith')

    parser.add_argument('--dataset_type',
                        metavar='dataset type',
                        type=str,
                        help='dataset type to work with')

    # Execute the parse_args() method
    args = parser.parse_args()

    arch_type = args.arch_type
    model_name = args.model_name
    K = args.K
    dataset = args.dataset
    dataset_type = args.dataset_type

    # Load the state dict of encoder
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    is_vae = False

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

    # Load the state dict of encoder
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()
    encoder.to(device)
    

    # Loads the embedding
    embedding = np.load(embedding_path)
    print(f'The Size of Embedding: {len(embedding)}')
    #print(embedding)

    indices_list = compute_similar_images(
            CONFIG.IMAGE_QUERY_PATH, K, embedding, device, is_vae
    )
    print(data_path)
    #plot_similar_images_(indices_list, data_path=data_path)
    compute_metrics(embedding, device, 10, 5, is_vae)