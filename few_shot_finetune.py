# Added new branch

import torch
torch.autograd.set_detect_anomaly(True)
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from FMoWDataset import FMoWDataset
from few_shot_dataset import create_few_shot_dataset, precompute_country_indices
from LinearClassifier import LinearClassifier
from CLIPModel import CLIPModel

def main():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model 
    model_name = 'RN50'
    checkpoint_path = '/deep/u/hellman1/MAML-Pytorch/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-RN50.pt'
    clip_model = CLIPModel(model_name, checkpoint_path)
    clip_model.model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    print("Loaded Model")
    
    # Load datasets from CSV files
    train_df = pd.read_csv('train_set.csv')
    val_df = pd.read_csv('val_set.csv')
    test_df = pd.read_csv('test_set.csv')  # If needed for evaluation
    print("Loaded Data")

    # Create datasets
    train_dataset = FMoWDataset(train_df, preprocess=clip_model.preprocess)
    val_dataset = FMoWDataset(val_df, preprocess=clip_model.preprocess)
    print("Datasets Created")

    num_classes = len(np.unique(train_df['country']))
    

    # Initialize the classifiers
    linear_classifier = LinearClassifier(feature_size=2048, num_classes=num_classes).to(device)
    print("Initialized LC")

    # Separate parameters into groups
    clip_params = {'params': clip_model.model.parameters(), 'lr': 0.001}  # Lower learning rate for CLIP
    classifier_params = {'params': linear_classifier.parameters(), 'lr': 0.01}  # Higher learning rate for Linear Classifier

    # Few-shot training settings
    shot_settings = [1, 4, 8, 16, 32]
    num_epochs = 1000
    batch_size = 10  # Adjust as needed
    num_way = 5

    # Pre-compute indices
    train_df = train_df.groupby('country')
    train_country_indices = precompute_country_indices(train_df)
    val_df = val_df.groupby('country')
    val_country_indices = precompute_country_indices(val_df)
    test_df = test_df.groupby('country')
    test_class_indices = precompute_country_indices(test_df)
    print("pre-computed class indices")

    for shots in shot_settings:
        print(f"Doing {shots}")
        optimizer = SGD([clip_params, classifier_params], weight_decay=4e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        initial_params = next(iter(clip_model.model.parameters())).clone()
        # print("Initial parameters:", initial_params)

        epoch = 0
        for epoch in tqdm(range(num_epochs), desc=f"Epoch {epoch+1}"):
            # print(f"On epoch {epoch}")
            few_shot_train_dataset = create_few_shot_dataset(train_dataset, shots, num_way, train_country_indices)
            # print("create_few_shot_dataset step passed")
            few_shot_train_loader = DataLoader(few_shot_train_dataset, batch_size=1, shuffle=True)
            # print("Ready to train")

            clip_model.train()
            linear_classifier.train()

            for images, texts, labels in few_shot_train_loader:
                # Reset gradients
                optimizer.zero_grad()

                # Tokenize the text
                text_inputs = tokenizer(texts).to(device)

                images_features = clip_model.encode_image(images)
                images_features_norm = images_features / images_features.norm(dim=-1, keepdim=True)

                text_features = clip_model.encode_text(text_inputs)
                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

                # Print out the shapes of the image and text features
                # print(f"image_features shape: {images_features_norm.shape}")
                # print(f"text_features shape: {text_features_norm.shape}")

                # Create targets for CosineEmbeddingLoss of 1 for all samples
                targets = torch.ones(images_features_norm.shape[0]).to(device)

                # Enforce consistency constraint
                criterion = nn.CosineEmbeddingLoss(reduction='mean')  # Set reduction to 'mean'
                consistency_loss = criterion(images_features_norm, text_features_norm, targets)

                # Linear Classifier Training
                combined_features = torch.cat((images_features_norm, text_features_norm), dim=1).float()
                outputs = linear_classifier(combined_features)
                classification_loss = F.cross_entropy(outputs, labels)

                # Total loss
                total_loss = consistency_loss + classification_loss
                # print(f"Total Loss {total_loss}")
                total_loss.backward()

                optimizer.step()

            scheduler.step()


        # After some training, check to see if clip model parameters have updated or not
        trained_params = next(iter(clip_model.model.parameters()))
        # print("Trained parameters:", trained_params)

        # Check if parameters have been updated
        if not torch.equal(initial_params, trained_params):
            print("Parameters have been updated.")
        else:
            print("Parameters have not been updated.")

        # Evaluate on validation set
        clip_model.eval()
        linear_classifier.eval()

        # create a few shot validation dataset and dataloader
        few_shot_val_dataset = create_few_shot_dataset(val_dataset, shots, num_way, val_country_indices)
        few_shot_val_loader = DataLoader(few_shot_val_dataset, batch_size=1, shuffle=True)

        linear_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, texts, labels in tqdm(few_shot_val_loader, desc="Evaluating on validation set"):
                # image_inputs = clip_model.preprocess_image(images)
                text_inputs = tokenizer(texts).to(device)

                # image_features = clip_model.encode_image(image_inputs).float()
                # text_features = clip_model.encode_text(text_inputs).float()

                images_features = clip_model.encode_image(images)
                images_features_norm = images_features / images_features.norm(dim=-1, keepdim=True)

                text_features = clip_model.encode_text(text_inputs)
                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)


                # Linear Classifier Prediction
                query_combined_features = torch.cat((images_features_norm, text_features_norm), dim=1)
                linear_outputs = linear_classifier(query_combined_features)
                linear_predictions = torch.argmax(linear_outputs, dim=1)
                linear_correct += (linear_predictions == labels).sum().item()

                total_samples += labels.size(0)

        linear_accuracy = 100 * linear_correct / total_samples
        print(f'Linear Classifier Accuracy for {shots} shots: {linear_accuracy}%')

    # Your existing code for saving results
    # ...

if __name__ == "__main__":
    main()
