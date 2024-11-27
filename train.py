import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules.update import RayEncoder
from modules.ldm import DepthPredictor, FusionBlock
from modules.img_encoder import ImageEncoder

from dataset import ... # to-do

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_dim = 128
num_epochs = 50
batch_size = 16
learning_rate = 1e-4
num_views = 8

class Quark(nn.Module):
    def __init__(self, feature_dim, num_views):
        super().__init__()
        self.feature_extractor = ImageEncoder()
        self.fusion_block = FusionBlock(feature_dim)
        self.depth_predictor = DepthPredictor(feature_dim)
        self.ray_encoder = RayEncoder()
    
    def forward(self, inputs):
        features = inputs["features"]
        rays = inputs["rays"]
        ray_features = self.ray_encoder(rays)
        extracted_features = self.feature_extractor(features)
        fused_features = self.fusion_block(extracted_features)
        depth = self.depth_predictor(fused_features + ray_features)
        return depth, fused_features

train_dataset = ...(split='train')
val_dataset = ...(split='val')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = Quark(feature_dim=feature_dim, num_views=num_views).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

def train_one_epoch(epoch, model, data_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch+1}"):
        features = batch["features"].to(device)
        rays = batch["rays"].to(device)
        target_depth = batch["depth"].to(device)
        inputs = {"features": features, "rays": rays}

        predicted_depth, _ = model(inputs)

        loss = criterion(predicted_depth, target_depth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# def validate_one_epoch(epoch, model, data_loader, criterion):
#     model.eval()
#     epoch_loss = 0
     
#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc=f"Validation Epoch {epoch+1}"):
#             features = batch["features"].to(device)
#             rays = batch["rays"].to(device)
#             target_depth = batch["depth"].to(device)
#             inputs = {"features": features, "rays": rays}
#             predicted_depth, _ = model(inputs)
#             loss = criterion(predicted_depth, target_depth)
#             epoch_loss += loss.item()
#     return epoch_loss / len(data_loader)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion)
    # val_loss = validate_one_epoch(epoch, model, val_loader, criterion)
    print(f"Epoch {epoch+1} / {num_epochs}, Train Loss: {train_loss:.4f}")    #, Val Loss: {val_loss:.4f}")
    
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"quark_epoch_{epoch + 1}.pth")