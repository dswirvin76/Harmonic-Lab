import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 128      # Increased for 47 classes
SEQ_LEN = 40 
BATCH_SIZE = 128
EPOCHS = 5            # EMNIST needs a bit more time
NUM_CLASSES = 47      # EMNIST Balanced Split
WEIGHTS_FILE = "harmonic_emnist.pth"

# --- Label Mapping (EMNIST Balanced) ---
# Maps index to character
EMNIST_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

# --- 1. The Harmonic Architecture ---
class HarmonicCell(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.base_freqs = nn.Parameter(torch.linspace(0.1, 12.0, hidden_dim), requires_grad=False)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.damping = nn.Parameter(torch.tensor(0.98))

    def step(self, h_prev, t, freq_mod, amp_mod):
        W_rec = self.A - self.A.T 
        effective_freqs = self.base_freqs + freq_mod
        osc = amp_mod * torch.sin(effective_freqs * t)
        h_next = (h_prev * self.damping) + 0.1 * (h_prev @ W_rec + osc)
        return torch.tanh(h_next)

class HarmonicEMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        flat_size = 64 * 7 * 7
        self.fc_init = nn.Linear(flat_size, LATENT_DIM)
        self.fc_freq = nn.Linear(flat_size, LATENT_DIM)
        self.fc_amp  = nn.Linear(flat_size, LATENT_DIM)
        
        self.cell = HarmonicCell(LATENT_DIM)
        self.classifier = nn.Linear(LATENT_DIM, NUM_CLASSES)

    def forward(self, x, return_dynamics=False):
        feats = self.encoder(x)
        h = torch.tanh(self.fc_init(feats))
        freq_mod = torch.tanh(self.fc_freq(feats))
        amp_mod = torch.sigmoid(self.fc_amp(feats))
        
        history = []
        energy_accum = torch.zeros_like(h)
        
        for t in range(SEQ_LEN):
            h = self.cell.step(h, t * 0.1, freq_mod, amp_mod)
            energy_accum += torch.abs(h)
            if return_dynamics:
                history.append(h)
                
        avg_energy = energy_accum / SEQ_LEN
        logits = self.classifier(avg_energy)
        
        if return_dynamics:
            return logits, torch.stack(history, dim=1)
        return logits

# --- 2. Training Logic ---
def train_emnist(model):
    print(f"--- Downloading and Training EMNIST (Balanced) on {DEVICE} ---")
    
    # EMNIST is rotated 90 degrees and flipped. We fix this transform.
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.transpose(1, 2) # Fix rotation
    ])
    
    train_data = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    model.train()
    
    total_steps = len(loader) * EPOCHS
    current_step = 0
    
    try:
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                output = model(data)
                loss = criterion(output, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_history.append(loss.item())
                current_step += 1
                
                if batch_idx % 50 == 0:
                    print(f"  Step {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}", end='\r')
            print("") # Newline
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current state.")

    # Save Model
    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"\n✅ Model saved to {WEIGHTS_FILE}")
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss', color='orange')
    plt.title("Harmonic Learning Curve (EMNIST)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# --- 3. Interactive Lab for Letters & Digits ---
class EMNISTLab:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('Harmonic EMNIST Laboratory')
        gs = self.fig.add_gridspec(2, 3)
        
        # Canvas
        self.ax_draw = self.fig.add_subplot(gs[:, 0])
        self.ax_draw.set_title("Draw Characters (0-9, A-Z, a-z)")
        self.canvas_grid = np.zeros((28, 28))
        self.img_plot = self.ax_draw.imshow(self.canvas_grid, cmap='gray', vmin=0, vmax=1)
        self.ax_draw.axis('off')
        
        # Waves
        self.ax_waves = self.fig.add_subplot(gs[0, 1:])
        self.ax_waves.set_title("Neural Harmonics")
        self.ax_waves.set_ylim(-1.5, 1.5)
        self.ax_waves.set_xlim(0, SEQ_LEN)
        self.waves = [self.ax_waves.plot([], [], lw=2)[0] for _ in range(5)]
        
        # Prediction Text (Instead of 47 bars, we just show top 3 text)
        self.ax_pred = self.fig.add_subplot(gs[1, 1:])
        self.ax_pred.axis('off')
        self.pred_text = self.ax_pred.text(0.5, 0.5, "Waiting...", ha='center', va='center', fontsize=24)

        self.drawing = False
        self.noise_level = 0.0
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        ax_clear = plt.axes([0.1, 0.05, 0.1, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self.clear_canvas)
        
        ax_noise_slider = plt.axes([0.3, 0.05, 0.4, 0.03])
        self.slider = Slider(ax_noise_slider, 'Noise Level', 0.0, 1.0, valinit=0.0)
        self.slider.on_changed(self.update_noise)

        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.run_inference)
        self.timer.start()
        
        plt.show()

    def on_click(self, event):
        if event.inaxes == self.ax_draw:
            self.drawing = True
            self.paint(event)

    def on_move(self, event):
        if self.drawing and event.inaxes == self.ax_draw:
            self.paint(event)

    def on_release(self, event):
        self.drawing = False

    def paint(self, event):
        if event.xdata is None: return
        x, y = int(event.xdata), int(event.ydata)
        r = 1
        for i in range(x-r, x+r+1):
            for j in range(y-r, y+r+1):
                if 0 <= i < 28 and 0 <= j < 28:
                    dist = np.sqrt((x-i)**2 + (y-j)**2)
                    intensity = max(0, 1 - dist/2)
                    self.canvas_grid[j, i] = min(1.0, self.canvas_grid[j, i] + intensity)
        self.img_plot.set_array(self.canvas_grid)
        self.fig.canvas.draw_idle()

    def clear_canvas(self, event):
        self.canvas_grid.fill(0)
        self.img_plot.set_array(self.canvas_grid)
        self.fig.canvas.draw_idle()

    def update_noise(self, val):
        self.noise_level = val

    def run_inference(self):
        img_tensor = torch.from_numpy(self.canvas_grid).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Add noise
        noise = torch.randn_like(img_tensor) * self.noise_level
        noisy_input = torch.clamp(img_tensor + noise, 0, 1)
        
        with torch.no_grad():
            logits, dynamics = self.model(noisy_input, return_dynamics=True)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            dyn_np = dynamics[0].cpu().numpy()
        
        # Get Top 3 Predictions
        top3_idx = np.argsort(probs)[-3:][::-1]
        
        res_str = f"Prediction: {EMNIST_MAPPING[top3_idx[0]]} ({probs[top3_idx[0]]:.2f})\n"
        res_str += f"2nd: {EMNIST_MAPPING[top3_idx[1]]} ({probs[top3_idx[1]]:.2f})  |  "
        res_str += f"3rd: {EMNIST_MAPPING[top3_idx[2]]} ({probs[top3_idx[2]]:.2f})"
        
        self.pred_text.set_text(res_str)
        if probs[top3_idx[0]] > 0.7:
            self.pred_text.set_color('green')
        else:
            self.pred_text.set_color('black')
            
        # Update Waves
        energies = np.sum(np.abs(dyn_np), axis=0)
        top_indices = np.argsort(energies)[-5:]
        t_axis = np.arange(SEQ_LEN)
        colors = plt.cm.jet(np.linspace(0, 1, 5))
        for i, idx in enumerate(top_indices):
            self.waves[i].set_data(t_axis, dyn_np[:, idx])
            self.waves[i].set_color(colors[i])
            
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    model = HarmonicEMNIST().to(DEVICE)
    
    if os.path.exists(WEIGHTS_FILE):
        print(f"--- Loading {WEIGHTS_FILE} ---")
        try:
            model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
            print("Loaded successfully.")
            EMNISTLab(model)
        except:
            print("Weight mismatch or corruption. Retraining.")
            train_emnist(model)
            EMNISTLab(model)
    else:
        train_emnist(model)
        EMNISTLab(model)