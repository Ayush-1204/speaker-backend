import argparse
import os
import random
from typing import Any
import logging
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# AM-Softmax Loss
# ---------------------------------------------------------
class AMSoftmaxLoss(nn.Module):
    """Additive Margin Softmax Loss for Speaker Verification."""
    def __init__(self, in_features, num_classes, margin=0.2, scale=30.0):
        super(AMSoftmaxLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(emb_norm, weight_norm)

        # Extract target logits
        target_logits = cosine[torch.arange(cosine.size(0)), labels]

        # Apply margin
        target_logits_with_margin = target_logits - self.margin

        # Put modified target logits back
        cosine_with_margin = cosine.clone()
        cosine_with_margin[torch.arange(cosine.size(0)), labels] = target_logits_with_margin

        # Scale
        scaled_logits = self.scale * cosine_with_margin
        
        # Cross entropy loss
        loss = F.cross_entropy(scaled_logits, labels)
        return loss

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class VoxCelebDataset(Dataset):
    def __init__(self, data_list_path, base_dir, chunk_len_sec=3.0, sr=16000):
        self.sr = sr
        self.chunk_samples = int(chunk_len_sec * sr)
        self.base_dir = Path(base_dir)
        self.data = []
        self.labels = []
        self.label_map = {}
        
        # Parse list: "speaker_id path/to/audio.wav"
        if not os.path.exists(data_list_path):
            logger.warning(f"Data list {data_list_path} not found. Running with dummy data.")
            self._setup_dummy()
            return
            
        with open(data_list_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    spk_id = parts[0]
                    audio_path = parts[1]
                    if spk_id not in self.label_map:
                        self.label_map[spk_id] = len(self.label_map)
                    self.data.append(audio_path)
                    self.labels.append(self.label_map[spk_id])
                    
    def _setup_dummy(self):
        self.label_map = {"dummy0": 0, "dummy1": 1}
        self.data = ["dummy_path_0", "dummy_path_1"] * 10
        self.labels = [0, 1] * 10
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.labels[index]
        path = self.data[index]
        
        if path.startswith("dummy_"):
            return torch.randn(self.chunk_samples), label
            
        try:
            full_path = self.base_dir / path
            info = torchaudio.info(full_path)
            num_frames = info.num_frames
            
            # Read random chunk
            if num_frames > self.chunk_samples:
                start_frame = random.randint(0, num_frames - self.chunk_samples)
                waveform, _ = torchaudio.load(full_path, frame_offset=start_frame, num_frames=self.chunk_samples)
            else:
                waveform, _ = torchaudio.load(full_path)
                # Pad if too short
                pad_size = self.chunk_samples - waveform.size(1)
                waveform = F.pad(waveform, (0, pad_size), "constant", 0)
                
            # Convert to mono if needed
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            return waveform.squeeze(0), label
        except Exception as e:
            # Fallback to zero tensor on read error
            logger.error(f"Error reading {path}: {e}")
            return torch.zeros(self.chunk_samples), label

# ---------------------------------------------------------
# Augmentation
# ---------------------------------------------------------
class AudioAugmenter(nn.Module):
    def __init__(self, musan_dir=None, rir_dir=None):
        super().__init__()
        self.musan_dir = musan_dir
        self.rir_dir = rir_dir
        self.snr_range = (5.0, 20.0)
        
    def forward(self, wavs):
        # Very basic augmentation wrapper. In a real system, you would load MUSAN audio 
        # and mix it at a random SNR, or apply RIR convolutions here.
        # Since this is a placeholder without the MUSAN corpus, we'll apply basic AWGN.
        batch_size = wavs.size(0)
        aug_wavs = []
        for i in range(batch_size):
            wav = wavs[i]
            if random.random() < 0.5:
                # Add additive white gaussian noise (simulating MUSAN/background)
                noise = torch.randn_like(wav)
                snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
                snr_linear = 10 ** (snr_db / 20.0)
                sig_rms = torch.sqrt(torch.mean(wav**2) + 1e-8)
                noise_rms = torch.sqrt(torch.mean(noise**2) + 1e-8)
                scaled_noise = noise * (sig_rms / (noise_rms * snr_linear))
                wav = wav + scaled_noise
            aug_wavs.append(wav)
        return torch.stack(aug_wavs)

# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="data/train_list.txt")
    parser.add_argument("--data_dir", type=str, default="data/wav")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--musan_dir", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="redimnet2_finetuned.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load pre-trained model
    logger.info("Loading pre-trained ReDimNet2 from Torch Hub...")
    try:
        model: Any = torch.hub.load("PalabraAI/redimnet2", "redimnet2", model_name="b6", train_type="pt_ft", dataset="vox2", pretrained=True, trust_repo=True)
    except Exception as e:
        logger.warning(f"Failed to load PalabraAI/redimnet2 ({e}). Falling back to IDRnD/ReDimNet...")
        model: Any = torch.hub.load("IDRnD/ReDimNet", "ReDimNet", trust_repo=True)
    
    model.to(device)
    model.train()

    # Freeze frontend (optional, but good for stability)
    # for param in model.feature_extractor.parameters():
    #     param.requires_grad = False

    # Setup data
    dataset = VoxCelebDataset(args.train_list, args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    num_classes = max(1, len(dataset.label_map))
    logger.info(f"Loaded {len(dataset)} samples across {num_classes} classes.")

    # Setup loss and optimizer
    # Embedding size for B6 is 192 (or 256 for other variants, check model config)
    # We can infer it by passing a dummy tensor
    with torch.no_grad():
        dummy_emb = model(torch.randn(1, 16000).to(device))
        emb_dim = dummy_emb.size(1)
    
    criterion = AMSoftmaxLoss(in_features=emb_dim, num_classes=num_classes).to(device)
    augmenter = AudioAugmenter(musan_dir=args.musan_dir).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': criterion.parameters()}
    ], lr=args.lr)

    # Train loop
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for wavs, labels in pbar:
            wavs = wavs.to(device)
            labels = labels.to(device)
            
            # Augment
            aug_wavs = augmenter(wavs)
            
            # Forward
            embeddings = model(aug_wavs)
            loss = criterion(embeddings, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, args.save_path)
        logger.info(f"Saved checkpoint to {args.save_path}")

if __name__ == "__main__":
    main()
