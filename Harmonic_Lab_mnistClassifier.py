import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import numpy as np
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 64
SEQ_LEN = 40 
BATCH_SIZE = 128 # Larger batch for faster training
EPOCHS = 8       # Enough for ~98% accuracy on full dataset
WEIGHTS_FILE = "harmonic_brain.pth"

# --- 1. The Harmonic Core ---
class HarmonicCell(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.base_freqs = nn.Parameter(torch.linspace(0.1, 10.0, hidden_dim), requires_grad=False)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.damping = nn.Parameter(torch.tensor(0.98))

    def step(self, h_prev, t, freq_mod, amp_mod):
        W_rec = self.A - self.A.T 
        effective_freqs = self.base_freqs + freq_mod
        osc = amp_mod * torch.sin(effective_freqs * t)
        h_next = (h_prev * self.damping) + 0.1 * (h_prev @ W_rec + osc)
        return torch.tanh(h_next)

class HarmonicNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        flat_size = 32 * 7 * 7
        self.fc_init = nn.Linear(flat_size, LATENT_DIM)
        self.fc_freq = nn.Linear(flat_size, LATENT_DIM)
        self.fc_amp  = nn.Linear(flat_size, LATENT_DIM)
        self.cell = HarmonicCell(LATENT_DIM)
        self.classifier = nn.Linear(LATENT_DIM, 10)

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

# --- 2. Training with Visualization ---
def train_visual(model):
    print("--- Phase 1: The Awakening (Full Dataset Training) ---")
    print("This will take a few minutes. Watch the neural patterns emerge...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    # FULL DATASET
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    plt.ion()
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    plt.subplots_adjust(bottom=0.2)
    
    axs[0].set_title("Input Reality")
    img_display = axs[0].imshow(np.zeros((28, 28)), cmap='gray')
    axs[0].axis('off')
    
    axs[1].set_title("Harmonic Energy Profile")
    bar_display = axs[1].bar(np.arange(LATENT_DIM), np.zeros(LATENT_DIM), color='cyan')
    axs[1].set_ylim(0, 1.5)
    
    axs[2].set_title("Learning Curve (Loss)")
    loss_line, = axs[2].plot([], [], 'r-', lw=2)
    loss_history = []
    
    model.train()
    
    step = 0
    try:
        for epoch in range(EPOCHS):
            print(f"--- Starting Epoch {epoch+1}/{EPOCHS} ---")
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                logits, dynamics = model(data, return_dynamics=True)
                loss = criterion(logits, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_history.append(loss.item())
                step += 1
                
                # Update visuals every 50 batches to keep training speed up
                if step % 50 == 0:
                    img_display.set_array(data[0, 0].cpu().numpy())
                    
                    # Visualize avg energy of the first item in batch
                    energy_profile = torch.mean(torch.abs(dynamics[0]), dim=0).detach().cpu().numpy()
                    for rect, h in zip(bar_display, energy_profile):
                        rect.set_height(h)
                    
                    # Update loss plot (downsampled to keep it fast)
                    if len(loss_history) > 0:
                        loss_data = loss_history[::10] # Plot every 10th point
                        loss_line.set_data(np.arange(len(loss_data)), loss_data)
                        axs[2].relim()
                        axs[2].autoscale_view()
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}", end='\r')
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")

    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"\n✅ Brain Saved to {WEIGHTS_FILE}")
    plt.ioff()
    plt.close()

# --- 3. Interactive Lab ---
class HarmonicLab:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title('Harmonic Resonance Laboratory')
        gs = self.fig.add_gridspec(2, 3)
        
        # Drawing Canvas
        self.ax_draw = self.fig.add_subplot(gs[:, 0])
        self.ax_draw.set_title("Draw Digit Here (Slowly)")
        self.canvas_grid = np.zeros((28, 28))
        self.img_plot = self.ax_draw.imshow(self.canvas_grid, cmap='gray', vmin=0, vmax=1)
        self.ax_draw.axis('off')
        
        # Waveforms
        self.ax_waves = self.fig.add_subplot(gs[0, 1:])
        self.ax_waves.set_title("Neural Resonance (Top 5 Fibers)")
        self.ax_waves.set_ylim(-1.1, 1.1)
        self.ax_waves.set_xlim(0, SEQ_LEN)
        self.waves = [self.ax_waves.plot([], [], lw=2)[0] for _ in range(5)]
        
        # Prediction
        self.ax_pred = self.fig.add_subplot(gs[1, 1])
        self.ax_pred.set_title("Class Probability")
        self.bars = self.ax_pred.bar(np.arange(10), np.zeros(10), color='lime')
        self.ax_pred.set_xticks(np.arange(10))
        self.ax_pred.set_ylim(0, 1)
        
        # Noise View
        self.ax_noise = self.fig.add_subplot(gs[1, 2])
        self.ax_noise.set_title("System Input (Noisy)")
        self.noisy_plot = self.ax_noise.imshow(np.zeros((28, 28)), cmap='gray', vmin=0, vmax=1)
        self.ax_noise.axis('off')

        self.drawing = False
        self.noise_level = 0.0
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        # UI
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
        if event.xdata is None or event.ydata is None: return
        x, y = int(event.xdata), int(event.ydata)
        # Brush Logic
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
        
        noise = torch.randn_like(img_tensor) * self.noise_level
        noisy_input = torch.clamp(img_tensor + noise, 0, 1)
        
        self.noisy_plot.set_array(noisy_input[0, 0].cpu().numpy())
        
        with torch.no_grad():
            logits, dynamics = self.model(noisy_input, return_dynamics=True)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            dyn_np = dynamics[0].cpu().numpy()
        
        # Color code bars
        max_idx = np.argmax(probs)
        for i, (bar, h) in enumerate(zip(self.bars, probs)):
            bar.set_height(h)
            if i == max_idx and h > 0.5:
                bar.set_color('#00ff00') # Bright Green
            else:
                bar.set_color('gray')
            
        # Update Waves
        energies = np.sum(np.abs(dyn_np), axis=0)
        top_indices = np.argsort(energies)[-5:] # Top 5 loudest neurons
        
        t_axis = np.arange(SEQ_LEN)
        colors = plt.cm.jet(np.linspace(0, 1, 5))
        for i, idx in enumerate(top_indices):
            self.waves[i].set_data(t_axis, dyn_np[:, idx])
            self.waves[i].set_color(colors[i])
            
        self.ax_waves.set_title(f"Neural Resonance (Active Fibers) | Prediction: {max_idx}")
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    model = HarmonicNetwork().to(DEVICE)
    
    # Check if we have a trained brain
    if os.path.exists(WEIGHTS_FILE):
        print(f"--- Loading existing brain from {WEIGHTS_FILE} ---")
        try:
            model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
            print("Brain loaded successfully.")
            HarmonicLab(model)
        except:
            print("Error loading brain. Retraining...")
            train_visual(model)
            HarmonicLab(model)
    else:
        # Train fresh
        train_visual(model)
        HarmonicLab(model)