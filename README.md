# Image-Similarity-Search
Simplified Image Similariy Search Engine using deep learning techniques

We used Autoencoders and variational autoencoders to generate embeddings which is further used to retrieve the images from the test set. The models are basedd on finetuning the encoder side with pretraiend resnet models and using simple decoder model for reconstruction. 

Dataset used: [Animals-10-Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
View the training logs here: [Training Logs](https://wandb.ai/ajwadakil/image-similarity-search?workspace=user-ajwadakil)

To run, specify the paths in the config file and then run infer.py for inference.
