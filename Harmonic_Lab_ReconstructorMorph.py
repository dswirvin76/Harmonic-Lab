import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import datasets, transforms
import os

# --- Configuration (MUST MATCH TRAINING) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 1024       # Matched to Harmonic_Autoencoder_CIFAR.py
SEQ_LEN = 40           # Matched to Harmonic_Autoencoder_CIFAR.py
WEIGHTS_FILE = "harmonic_ae_cifar.pth"

# --- 1. The Architecture (Restored from Training Script) ---
class HarmonicCell(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.base_freqs = nn.Parameter(torch.linspace(0.01, 12.0, hidden_dim), requires_grad=False)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.damping = nn.Parameter(torch.tensor(0.98)) 

    def step(self, h_prev, t, freq_mod, amp_mod):
        W_rec = self.A - self.A.T 
        effective_freqs = self.base_freqs + freq_mod
        osc = amp_mod * torch.sin(effective_freqs * t)
        h_next = (h_prev * self.damping) + 0.1 * (h_prev @ W_rec + osc)
        return torch.tanh(h_next)

class HarmonicAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        flat_size = 128 * 4 * 4
        
        # Physics Parameters
        self.fc_init = nn.Linear(flat_size, LATENT_DIM)
        self.fc_freq = nn.Linear(flat_size, LATENT_DIM)
        self.fc_amp  = nn.Linear(flat_size, LATENT_DIM)
        
        self.cell = HarmonicCell(LATENT_DIM)
        
        # Decoder
        self.decoder_input = nn.Linear(LATENT_DIM, flat_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    # We need a special forward pass to handle manual parameters
    def manual_forward(self, h, freq_mod, amp_mod):
        # Run dynamics with provided physics parameters
        curr_h = h
        for t in range(SEQ_LEN):
            curr_h = self.cell.step(curr_h, t * 0.1, freq_mod, amp_mod)
        
        # Decode the final state
        return self.decoder(self.decoder_input(curr_h))

# --- 2. The Jam Session Logic ---

def get_physics_params(model, img):
    """Extracts the harmonic parameters (h, freq, amp) from an image."""
    with torch.no_grad():
        feats = model.encoder(img)
        h = torch.tanh(model.fc_init(feats))
        freq = torch.tanh(model.fc_freq(feats))
        amp = torch.sigmoid(model.fc_amp(feats))
    return h, freq, amp

def interpolate(p1, p2, steps):
    """Linearly interpolates between two sets of tensors."""
    alphas = np.linspace(0, 1, steps)
    interps = []
    for a in alphas:
        # Linear blend: (1-a)*start + a*end
        val = (1 - a) * p1 + a * p2
        interps.append(val)
    return interps

def run_jam_session():
    if not os.path.exists(WEIGHTS_FILE):
        print(f"❌ Error: {WEIGHTS_FILE} not found. Please train the Harmonic_Autoencoder_CIFAR.py first.")
        return

    print(f"--- Loading Harmonic Resonator from {WEIGHTS_FILE} ---")
    model = HarmonicAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
    model.eval()
    
    # Load Data
    dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
    
    # --- Pick Two Images to Morph ---
    # Index 8 = Ship, Index 11 = Truck, Index 3 = Airplane
    # Feel free to change these indices to mix different concepts
    idx_A = 36  
    idx_B = 12 
    
    img_A_raw = dataset[idx_A][0]
    img_B_raw = dataset[idx_B][0]
    
    img_A = img_A_raw.unsqueeze(0).to(DEVICE)
    img_B = img_B_raw.unsqueeze(0).to(DEVICE)
    
    print("Extracting Harmonic Signatures...")
    h1, f1, a1 = get_physics_params(model, img_A)
    h2, f2, a2 = get_physics_params(model, img_B)
    
    print("Simulating Harmonic Interference (Morphing)...")
    
    # Create Morph Sequence
    DURATION_FRAMES = 60
    
    h_seq = interpolate(h1, h2, DURATION_FRAMES)
    f_seq = interpolate(f1, f2, DURATION_FRAMES)
    a_seq = interpolate(a1, a2, DURATION_FRAMES)
    
    generated_frames = []
    
    with torch.no_grad():
        for i in range(DURATION_FRAMES):
            # For every frame, we run the FULL physics simulation
            # using the blended parameters
            recon = model.manual_forward(h_seq[i], f_seq[i], a_seq[i])
            
            # Convert to numpy image
            np_img = recon[0].cpu().permute(1, 2, 0).numpy()
            generated_frames.append(np_img)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Show Anchors
    axes[0].imshow(img_A_raw.permute(1, 2, 0))
    axes[0].set_title("Source (A)")
    axes[0].axis('off')
    
    axes[2].imshow(img_B_raw.permute(1, 2, 0))
    axes[2].set_title("Target (B)")
    axes[2].axis('off')
    
    # Animate Middle
    axes[1].axis('off')
    axes[1].set_title("Morph")
    im_display = axes[1].imshow(generated_frames[0])
    
    def update(frame_idx):
        im_display.set_data(generated_frames[frame_idx])
        return [im_display]

    ani = animation.FuncAnimation(fig, update, frames=len(generated_frames), interval=50, blit=True)
    
    print("Displaying Jam Session...")
    plt.show()

if __name__ == "__main__":
    run_jam_session()