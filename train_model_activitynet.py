import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import json

class VideoDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.keys = list(data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        video_id = self.keys[idx]
        item = self.data[video_id]
        return item['sentences'], item['timestamps']

def collate_fn(batch):
    sentences, timestamps = zip(*batch)

    # Find the length of the longest sequence
    max_len_sentences = max(len(s) for s in sentences)
    max_len_timestamps = max(len(t) for t in timestamps)

    # Pad sequences to the same length
    padded_sentences = []
    for s in sentences:
        padded_sentences.append(s + [''] * (max_len_sentences - len(s)))

    padded_timestamps = []
    for t in timestamps:
        padded_timestamps.append(t + [[0, 0]] * (max_len_timestamps - len(t)))

    return padded_sentences, padded_timestamps

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.video_encoder = nn.Linear(2048, 512)  # Example dimensions
        self.audio_encoder = nn.Linear(128, 512)  # Example dimensions
        self.text_encoder = nn.Embedding(10000, 512)  # Example dimensions
        self.fc = nn.Linear(1536, 512)  # Adjust to match GRU input size
        self.fc_out = nn.Linear(512, 2)  # To match target shape
        self.decoder = nn.GRU(512, 512, num_layers=2, batch_first=True)

    def forward(self, video_features, audio_features, text_features):
        video_emb = self.video_encoder(video_features)
        audio_emb = self.audio_encoder(audio_features)
        
        # Sum text embeddings across the sequence length dimension to reduce dimensionality
        text_emb = self.text_encoder(text_features).sum(dim=1)

        # Ensure all embeddings have the same number of dimensions
        combined_features = torch.cat((video_emb, audio_emb, text_emb), dim=1)
        combined_features = self.fc(combined_features)
        output, _ = self.decoder(combined_features.unsqueeze(1))
        output = self.fc_out(output.squeeze(1))
        return output

def train_model(model, dataloader, epochs=5):
    criterion = nn.MSELoss()  # Use Mean Squared Error Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for sentences, timestamps in dataloader:
            optimizer.zero_grad()
            video_features = torch.randn(len(sentences), 2048)  # Example random tensor
            audio_features = torch.randn(len(sentences), 128)  # Example random tensor
            text_features = torch.randint(0, 10000, (len(sentences), 10))  # Example random tensor
            output = model(video_features, audio_features, text_features)
            
            # Flatten timestamps for MSELoss
            target = torch.tensor(timestamps).float().view(len(sentences), -1, 2)  # Ensure it's a float tensor
            
            # Ensure the output and target tensors have the same shape
            if output.shape != target.shape:
                target = target[:, :output.size(1), :]
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def load_activitynet_data(dataset_dir):
    data = {}
    files = ["train.json", "val_1.json", "val_2.json"]
    for file in files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                print(f"Loaded {len(file_data)} items from {file}")
                data.update(file_data)
    return data

data = load_activitynet_data("captions")
dataset = VideoDataset(data)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
model = MultimodalModel()
train_model(model, dataloader)
