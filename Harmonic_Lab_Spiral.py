import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_POINTS = 400         # Points per class
NOISE = 0.05           # Fuzziness
SEQ_LEN = 1           # How long the neurons "spin" before deciding
LATENT_DIM = 36       # Number of harmonic neurons
EPOCHS = 8000

# --- 1. Data Generation (The Spiral) ---
def generate_spiral_data(n_points, noise):
    X = []
    y = []
    
    # Class 0 (Blue)
    for i in range(n_points):
        r = i / n_points * 5 + 0.2
        t = 1.75 * i / n_points * 2 * np.pi + 0.5
        xi = r * np.sin(t) + np.random.randn() * noise
        yi = r * np.cos(t) + np.random.randn() * noise
        X.append([xi, yi])
        y.append(0)
        
    # Class 1 (Orange)
    for i in range(n_points):
        r = i / n_points * 5 + 0.2
        t = 1.75 * i / n_points * 2 * np.pi + np.pi + 0.5 # 180 deg offset
        xi = r * np.sin(t) + np.random.randn() * noise
        yi = r * np.cos(t) + np.random.randn() * noise
        X.append([xi, yi])
        y.append(1)
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# --- 2. The Harmonic Architecture ---
class HarmonicCell(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Initial frequencies spread out to capture different rotations
        self.base_freqs = nn.Parameter(torch.linspace(0.1, 5.0, hidden_dim), requires_grad=False)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.damping = nn.Parameter(torch.tensor(0.99))

    def step(self, h_prev, t, freq_mod, amp_mod):
        # Rotation Matrix (Skew-Symmetric)
        W_rec = self.A - self.A.T 
        
        # The Driver
        osc = amp_mod * torch.sin((self.base_freqs + freq_mod) * t)
        
        # Euler Integration
        h_next = (h_prev * self.damping) + 0.1 * (h_prev @ W_rec + osc)
        return torch.tanh(h_next)

class HarmonicSpiralClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Input is just x, y coordinates
        self.encoder = nn.Linear(2, 32)
        
        # Map (x,y) to physics parameters
        self.fc_init = nn.Linear(32, LATENT_DIM)
        self.fc_freq = nn.Linear(32, LATENT_DIM)
        self.fc_amp  = nn.Linear(32, LATENT_DIM)
        
        self.cell = HarmonicCell(LATENT_DIM)
        self.classifier = nn.Linear(LATENT_DIM, 1) # Binary Output

    def forward(self, x):
        # 1. Encode Position
        feats = torch.tanh(self.encoder(x))
        
        # 2. Set Physics
        h = torch.tanh(self.fc_init(feats))
        freq = torch.tanh(self.fc_freq(feats))
        amp = torch.sigmoid(self.fc_amp(feats))
        
        # 3. Resonate
        # This is where the magic happens. 
        # Points in the spiral are converted to frequencies.
        # After T steps, points at different positions will be at different phases.
        for t in range(SEQ_LEN):
            h = self.cell.step(h, t * 0.2, freq, amp)
            
        # 4. Classify based on final energy state
        return self.classifier(h)

# --- 3. The Visual Lab ---
def train_spiral():
    # Setup Data
    X_train, y_train = generate_spiral_data(N_POINTS, NOISE)
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    
    model = HarmonicSpiralClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # Plot Setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title('Harmonic Spiral Test')
    
    # Create Grid for Contour Plot (The "Background" colors)
    xx, yy = np.meshgrid(np.arange(-6, 6, 0.1), np.arange(-6, 6, 0.1))
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    
    # Colors (Matching TF Playground)
    # Class 0: Blue, Class 1: Orange
    cm = plt.cm.RdBu_r
    cm_bright = ListedColormap(['#0000FF', '#FF0000'])
    
    print("--- Starting Harmonic Resonance Training ---")
    
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Visual Update every 10 epochs
        if epoch % 10 == 0:
            ax.clear()
            
            # 1. Run Model on Background Grid
            model.eval()
            with torch.no_grad():
                grid_logits = model(grid_tensor)
                grid_probs = torch.sigmoid(grid_logits).reshape(xx.shape).cpu().numpy()
            
            # 2. Draw Contour (Decision Boundary)
            ax.contourf(xx, yy, grid_probs, cmap=cm, alpha=0.8)
            
            # 3. Draw Data Points
            y_np = y_train.cpu().numpy().flatten()
            X_np = X_train.cpu().numpy()
            
            ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap=cm_bright, edgecolors='k', s=40)
            
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_title(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.pause(0.01)
            
            if loss.item() < 0.0001:
                print("Converged!")
                break
                
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train_spiral()