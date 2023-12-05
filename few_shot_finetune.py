import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from FMoWDataset import FMoWDataset
from few_shot_dataset import create_few_shot_dataset, precompute_class_indices
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
    print("Loaded Model")
    
    # Load dataset
    df = pd.read_csv('final.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[:10000]
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = FMoWDataset(train_df, preprocess=clip_model.preprocess_image)
    val_dataset = FMoWDataset(val_df, preprocess=clip_model.preprocess_image)
    print("Loaded Data")

    num_classes = len(np.unique(train_df['country']))

    # Initialize the classifiers
    linear_classifier = LinearClassifier(feature_size=1024, num_classes=num_classes).to(device)
    print("Initialized LC")

    # Few-shot training settings
    shot_settings = [1, 4, 8, 16, 32]
    num_epochs = 10
    batch_size = 10  # Adjust as needed

    # Pre-compute indices
    class_indices = precompute_class_indices(train_dataset)
    print("pre-computed")

    for shots in shot_settings:
        print(f"Doing {shots}")
        optimizer = SGD(linear_classifier.parameters(), lr=0.8, weight_decay=4e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            print(f"On epoch {epoch}")
            few_shot_train_dataset = create_few_shot_dataset(train_dataset, shots, num_classes, class_indices)
            print("create_few_shot_dataset step passed")
            few_shot_train_loader = DataLoader(few_shot_train_dataset, batch_size=batch_size, shuffle=True)
            print("Ready to train")

            for images, texts, labels in tqdm(few_shot_train_loader, desc=f"Training with {shots} shots"):
                images, texts, labels = images.to(device), texts.to(device), labels.to(device)
                # Reset gradients
                optimizer.zero_grad()

                # Process support set
                image_inputs = clip_model.preprocess_image(images)
                text_inputs = clip_model.tokenize_texts(texts)
                image_features = clip_model.encode_image(image_inputs).float()
                text_features = clip_model.encode_text(text_inputs).float()

                # Enforce consistency constraint
                consistency_loss = contrastive_loss(image_features, text_features)

                # Linear Classifier Training
                combined_features = torch.cat((image_features, text_features), dim=1)
                outputs = linear_classifier(combined_features)
                classification_loss = cross_entropy_loss(outputs, labels)

                # Total loss
                total_loss = consistency_loss + classification_loss
                print(f"Total Loss {total_loss}")
                total_loss.backward()

                optimizer.step()

            scheduler.step()

        # Evaluate on validation set
        clip_model.eval()
        linear_classifier.eval()
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        linear_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, texts, labels in tqdm(val_dataloader, desc="Evaluating on validation set"):
                image_inputs = clip_model.preprocess_image(images)
                text_inputs = clip_model.tokenize_texts(texts)
                image_features = clip_model.encode_image(image_inputs).float()
                text_features = clip_model.encode_text(text_inputs).float()

                # Linear Classifier Prediction
                query_combined_features = torch.cat((image_features, text_features), dim=1)
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
