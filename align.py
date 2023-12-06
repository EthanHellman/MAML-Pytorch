import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import open_clip

from FMoWDataset import FMoWDataset
from few_shot_dataset import create_few_shot_dataset, precompute_country_indices
from LinearClassifier import LinearClassifier
from CLIPModel import CLIPModel

def info_nce_loss(image_features, text_features, temperature):
    # Normalize the features
    image_features_norm = F.normalize(image_features, dim=1)
    text_features_norm = F.normalize(text_features, dim=1)
    
    # Calculate the logits
    logits_per_image = torch.matmul(image_features_norm, text_features_norm.T) / temperature
    logits_per_text = logits_per_image.T
    
    # Labels are the diagonal of the similarity matrix
    labels = torch.arange(image_features.size(0), device=image_features.device)
    
    # Calculate the InfoNCE loss
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    
    # Average the two losses
    total_loss = (loss_i2t + loss_t2i) / 2
    return total_loss

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
    train_df = pd.read_csv('train_align.csv')
    val_df = pd.read_csv('val_align.csv')
    # If test set is needed for evaluation, load it here
    print("Loaded Data")

    # Create datasets
    train_dataset = FMoWDataset(train_df, preprocess=clip_model.preprocess)
    val_dataset = FMoWDataset(val_df, preprocess=clip_model.preprocess)
    print("Datasets Created")

    # Create batch dataloaders for the train and val datasets
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Dataloaders Created")

    # Initialize the optimizer
    optimizer = SGD(clip_model.model.parameters(), lr=0.01, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)

    # Temperature parameter for InfoNCE loss, needs to be tuned
    temperature = 0.07
    num_epochs = 10  # Define the number of epochs

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        clip_model.train()
        total_loss = 0
        for images, texts, _ in train_loader:
            optimizer.zero_grad()  # Reset gradients

            # Tokenize the text
            text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            # images = images.to(device)

            # Forward pass through CLIP model
            image_features = clip_model.encode_image(images)
            text_features = clip_model.encode_text(text_inputs.input_ids)

            # Calculate InfoNCE Loss
            nce_loss = info_nce_loss(image_features, text_features, temperature)
            total_loss += nce_loss.item()
            
            nce_loss.backward()
            optimizer.step()

        scheduler.step()  # Update the learning rate

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validation logic
        clip_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, texts, _ in val_loader:
                # Tokenize the text
                text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
                images = images.to(device)

                # Forward pass through CLIP model
                image_features = clip_model.encode_image(images)
                text_features = clip_model.encode_text(text_inputs.input_ids)

                # Calculate InfoNCE Loss
                nce_loss = info_nce_loss(image_features, text_features, temperature)
                total_val_loss += nce_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    main()
