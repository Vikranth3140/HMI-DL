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

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.video_encoder = nn.Linear(2048, 512)  # Example dimensions
        self.audio_encoder = nn.Linear(128, 512)  # Example dimensions
        self.text_encoder = nn.Embedding(10000, 512)  # Example dimensions
        self.fc = nn.Linear(1536, 512)
        self.decoder = nn.GRU(512, 512, num_layers=2, batch_first=True)

    def forward(self, video_features, audio_features, text_features):
        video_emb = self.video_encoder(video_features)
        audio_emb = self.audio_encoder(audio_features)
        text_emb = self.text_encoder(text_features)
        combined_features = torch.cat((video_emb, audio_emb, text_emb), dim=1)
        combined_features = self.fc(combined_features)
        output, _ = self.decoder(combined_features)
        return output

def train_model(model, dataloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for sentences, timestamps in dataloader:
            optimizer.zero_grad()
            video_features = torch.randn(len(sentences), 2048)  # Example random tensor
            audio_features = torch.randn(len(sentences), 128)  # Example random tensor
            text_features = torch.randint(0, 10000, (len(sentences), 10))  # Example random tensor
            output = model(video_features, audio_features, text_features)
            loss = criterion(output, torch.tensor(timestamps))
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

data = load_activitynet_data("/mnt/data")
dataset = VideoDataset(data)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
model = MultimodalModel()
train_model(model, dataloader)