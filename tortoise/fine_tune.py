import torch
import torchaudio
import os
from torch.utils.data import DataLoader, Dataset
# from tortoise.models.diffusion_decoder import DiffusionTTS
# from tortoise.models.autoregressive import Autoregressive
# from tortoise.utils.tokenizer import VoiceBpeTokenizer
from transformers import AutoTokenizer
import torchaudio.transforms as T

tokenizer_ar = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-da")
# Define dataset loader
class ArabicTTSDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, tokenizer, max_length=128, mel_spec_params=None):
        self.data = []
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Define default mel spectrogram parameters if not provided
        self.mel_spec_params = mel_spec_params or {
            "n_mels": 80,
            "n_fft": 1024,
            "hop_length": 256
        }

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                audio_file, text = line.strip().split("|")
                self.data.append((audio_file, text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, text = self.data[idx]

        # Load audio
        waveform, sr = torchaudio.load(os.path.join(self.audio_dir, audio_path))

        # Convert to mel spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_mels=self.mel_spec_params["n_mels"],
            n_fft=self.mel_spec_params["n_fft"],
            hop_length=self.mel_spec_params["hop_length"]
        )
        mel_spec = mel_transform(waveform).squeeze(0)  # Remove batch dimension

        # Normalize the mel spectrogram
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        # Tokenize and pad text
        tokenized_text = self.tokenizer.encode(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return tokenized_text.squeeze(0), mel_spec  # Remove batch dimension

# Load dataset
dataset = ArabicTTSDataset("arabic-speech-corpus/arabic-speech-corpus/orthographic-transcript.csv", "arabic-speech-corpus/arabic-speech-corpus/wav", tokenizer_ar)

train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load pre-trained model
autoregressive_model = torch.load("models/autoregressive.pth")
diffusion_model = torch.load("models/diffusion_decoder.pth")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoregressive_model.to(device)
diffusion_model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(
    list(autoregressive_model.parameters()) + list(diffusion_model.parameters()),  # Optimize both models
    lr=1e-4
)
criterion = torch.nn.L1Loss()  # MAE loss is better for spectrograms

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0  

    for text_tokens, mel_spec in train_loader:
        text_tokens, mel_spec = text_tokens.to(device), mel_spec.to(device)

        optimizer.zero_grad()

        # Step 1: Autoregressive model predicts mel spectrogram
        predicted_mel = autoregressive_model(text_tokens.unsqueeze(0))

        # Step 2: Diffusion model refines the predicted mel spectrogram into final output
        predicted_audio = diffusion_model(predicted_mel)

        # Step 3: Compute loss (L1 loss between generated audio and ground truth spectrogram)
        loss = criterion(predicted_audio, mel_spec)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Save the **entire model** instead of state_dict()
torch.save(autoregressive_model, "models/autoregressive_finetuned.pth")
torch.save(diffusion_model, "models/diffusion_decoder_finetuned.pth")

print("Fine-tuned models saved successfully!")
