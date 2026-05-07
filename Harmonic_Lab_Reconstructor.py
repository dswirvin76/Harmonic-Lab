import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 1024       # Increased capacity for pixel reconstruction
SEQ_LEN = 40           # Duration of the harmonic "thought" process
BATCH_SIZE = 64
EPOCHS = 48
WEIGHTS_FILE = "harmonic_ae_cifar.pth"

# --- 1. The Harmonic Architecture ---
class HarmonicCell(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Frequencies initialized to cover low (structure) and high (texture) ranges
        self.base_freqs = nn.Parameter(torch.linspace(0.01, 12.0, hidden_dim), requires_grad=False)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.damping = nn.Parameter(torch.tensor(0.999)) 

    def step(self, h_prev, t, freq_mod, amp_mod):
        # Skew-symmetric matrix for rotation
        W_rec = self.A - self.A.T 
        effective_freqs = self.base_freqs + freq_mod
        
        # Drive the oscillator
        osc = amp_mod * torch.sin(effective_freqs * t)
        
        # Integration step
        h_next = (h_prev * self.damping) + 0.1 * (h_prev @ W_rec + osc)
        return torch.tanh(h_next)

class HarmonicAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- ENCODER (Compress Image to Latent Physics) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(), # 16x16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), # 8x8
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), # 4x4
            nn.Flatten()
        )
        
        flat_size = 128 * 4 * 4
        
        # Physics Parameters projections
        self.fc_init = nn.Linear(flat_size, LATENT_DIM)
        self.fc_freq = nn.Linear(flat_size, LATENT_DIM)
        self.fc_amp  = nn.Linear(flat_size, LATENT_DIM)
        
        self.cell = HarmonicCell(LATENT_DIM)
        
        # --- DECODER (Reconstruct Image from Latent State) ---
        self.decoder_input = nn.Linear(LATENT_DIM, flat_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(), # 8x8
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(), # 16x16
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid() # Force output to [0, 1] range
        )

    def forward(self, x, return_sequence=False):
        # 1. Encode
        feats = self.encoder(x)
        h = torch.tanh(self.fc_init(feats))
        freq_mod = torch.tanh(self.fc_freq(feats))
        amp_mod = torch.sigmoid(self.fc_amp(feats))
        
        # 2. Resonate
        history_h = []
        for t in range(SEQ_LEN):
            h = self.cell.step(h, t * 0.1, freq_mod, amp_mod)
            if return_sequence:
                history_h.append(h)
        
        # 3. Decode
        if return_sequence:
            # Decode every single time step to make a video
            # Stack: (Seq, Batch, Latent)
            stacked_h = torch.stack(history_h, dim=0) 
            seq_len, batch, lat = stacked_h.shape
            
            # Flatten to pass through decoder in one go
            # (Seq*Batch, Latent)
            flat_h = stacked_h.view(-1, lat)
            
            # Project and Decode
            decoded_flat = self.decoder(self.decoder_input(flat_h))
            
            # Reshape back to (Seq, Batch, C, H, W)
            decoded_seq = decoded_flat.view(seq_len, batch, 3, 32, 32)
            return decoded_seq, stacked_h
            
        else:
            # Standard training: just decode the final state
            recon = self.decoder(self.decoder_input(h))
            return recon

# --- 2. Training ---
def train_autoencoder(model):
    print("--- Initializing Harmonic Autoencoder ---")
    
    # Simple 0-1 Normalization for easier reconstruction visualization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
    # Mean Squared Error for reconstruction
    criterion = nn.MSELoss() 
    
    model.train()
    loss_history = []
    
    print(f"Training on {DEVICE} for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (data, _) in enumerate(loader):
            data = data.to(DEVICE) # Ignore labels
            
            optimizer.zero_grad()
            
            # Forward pass (Standard)
            recon = model(data)
            
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Step {i} | MSE Loss: {loss.item():.5f}", end='\r')
        
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"\nEpoch {epoch+1} Complete. Avg Loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"✅ Weights saved to {WEIGHTS_FILE}")
    
    plt.plot(loss_history)
    plt.title("Reconstruction Convergence")
    plt.show()

# --- 3. The Reconstruction Lab ---
class ReconstructionLab:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        self.dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.iter_loader = iter(self.loader)
        
        # State storage
        self.original_img = None
        self.recon_seq = None  # The movie of reconstruction
        self.dynamics_seq = None # The latent waves
        
        # GUI Setup
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('Harmonic Reconstruction Lab')
        gs = self.fig.add_gridspec(2, 3)
        
        # Original Image
        self.ax_orig = self.fig.add_subplot(gs[0, 0])
        self.ax_orig.set_title("Input")
        self.ax_orig.axis('off')
        self.plot_orig = self.ax_orig.imshow(np.zeros((32,32,3)))
        
        # Reconstructed Image
        self.ax_recon = self.fig.add_subplot(gs[0, 1])
        self.ax_recon.set_title("Reconstruction (t=0)")
        self.ax_recon.axis('off')
        self.plot_recon = self.ax_recon.imshow(np.zeros((32,32,3)))
        
        # Latent Dynamics Plot
        self.ax_wave = self.fig.add_subplot(gs[1, :])
        self.ax_wave.set_title("Internal Harmonic Resonance")
        self.ax_wave.set_ylim(-1.5, 1.5)
        self.ax_wave.set_xlim(0, SEQ_LEN)
        self.lines = [self.ax_wave.plot([], [])[0] for _ in range(10)] # Show top 10 frequencies
        self.time_line = self.ax_wave.axvline(0, color='red', linestyle='--')

        # Controls
        ax_time = plt.axes([0.2, 0.02, 0.6, 0.03])
        self.s_time = Slider(ax_time, 'Time Step', 0, SEQ_LEN-1, valinit=SEQ_LEN-1, valstep=1)
        self.s_time.on_changed(self.update_frame)
        
        ax_next = plt.axes([0.85, 0.02, 0.1, 0.05])
        self.btn_next = Button(ax_next, 'Next Img')
        self.btn_next.on_clicked(self.next_image)
        
        self.next_image(None)
        plt.show()

    def next_image(self, event):
        try:
            img, _ = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.loader)
            img, _ = next(self.iter_loader)
            
        self.original_img = img.permute(0, 2, 3, 1).numpy()[0]
        
        # Run Inference with sequence return
        with torch.no_grad():
            # recon_seq shape: (Seq, 1, 3, 32, 32)
            # dynamics_seq shape: (Seq, 1, Latent)
            r_seq, d_seq = self.model(img.to(DEVICE), return_sequence=True)
            
            self.recon_seq = r_seq.cpu().permute(0, 1, 3, 4, 2).numpy()[:, 0, :, :, :]
            self.dynamics_seq = d_seq.cpu().numpy()[:, 0, :]
            
        # Update original plot
        self.plot_orig.set_data(self.original_img)
        
        # Update waves background
        t = np.arange(SEQ_LEN)
        # Pick top 10 most energetic neurons to display
        energies = np.sum(np.abs(self.dynamics_seq), axis=0)
        top_indices = np.argsort(energies)[-10:]
        
        for i, idx in enumerate(top_indices):
            self.lines[i].set_data(t, self.dynamics_seq[:, idx])
            
        self.s_time.set_val(SEQ_LEN-1) # Jump to end
        self.update_frame(SEQ_LEN-1)

    def update_frame(self, val):
        t = int(val)
        if self.recon_seq is None: return
        
        # Update Reconstruction Image
        img_t = self.recon_seq[t]
        img_t = np.clip(img_t, 0, 1)
        self.plot_recon.set_data(img_t)
        self.ax_recon.set_title(f"Reconstruction (t={t})")
        
        # Update Timeline Marker
        self.time_line.set_xdata([t])
        
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    model = HarmonicAutoencoder().to(DEVICE)
    
    if os.path.exists(WEIGHTS_FILE):
        print(f"--- Loading Weights: {WEIGHTS_FILE} ---")
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
        ReconstructionLab(model)
    else:
        train_autoencoder(model)
        ReconstructionLab(model)