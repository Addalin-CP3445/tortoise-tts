import torch
import torchaudio
import os
from torch.utils.data import DataLoader, Dataset
from tortoise.models.diffusion_decoder import DiffusionTTS
from tortoise.models.autoregressive import Autoregressive
from tortoise.utils.tokenizer import VoiceBpeTokenizer

# Define dataset loader
class ArabicTTSDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, tokenizer):
        self.data = []
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                audio_file, text = line.strip().split("|")
                self.data.append((audio_file, text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, text = self.data[idx]
        audio_tensor, sr = torchaudio.load(os.path.join(self.audio_dir, audio_path))
        text_tokens = self.tokenizer.encode(text)
        return text_tokens, audio_tensor

# Load dataset
dataset = ArabicTTSDataset("arabic-speech-corpus/arabic-speech-corpus/orthographic-transcript.csv", "arabic-speech-corpus/arabic-speech-corpus/wav", VoiceBpeTokenizer())

train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load pre-trained model
autoregressive_model = Autoregressive()
diffusion_model = DiffusionTTS()

autoregressive_model.load_state_dict(torch.load("models/autoregressive.pth"))
diffusion_model.load_state_dict(torch.load("models/diffusion_decoder.pth"))

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoregressive_model.to(device)
diffusion_model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(autoregressive_model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for text_tokens, audio_tensor in train_loader:
        text_tokens, audio_tensor = text_tokens.to(device), audio_tensor.to(device)
        
        optimizer.zero_grad()
        output = autoregressive_model(text_tokens)
        loss = criterion(output, audio_tensor)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save fine-tuned model
torch.save(autoregressive_model.state_dict(), "models/autoregressive_finetuned.pth")
