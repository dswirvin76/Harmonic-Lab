import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# --- HYPERPARAMETERS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_SIZE = 5                
SPACING = 1.0                
DT = 0.2                     
LEARNING_RATE = 0.005
SEQ_LEN = 12                 

IMG_SIZE = 12
AUD_BINS = 12
SAVE_PATH = "toddler_brain.pth"

# --- 1. THE TODDLER'S WORLD ---
def generate_bouncing_ball():
    v, a = torch.zeros((SEQ_LEN, IMG_SIZE, IMG_SIZE)), torch.zeros((SEQ_LEN, AUD_BINS))
    for t in range(SEQ_LEN):
        y = int(abs(np.sin(t * 0.5)) * (IMG_SIZE - 2))
        v[t, y:y+2, 5:7] = 1.0
        if y > IMG_SIZE - 4: a[t, 1:4] = 1.0 
    return v.view(SEQ_LEN, -1), a

def generate_passing_train():
    v, a = torch.zeros((SEQ_LEN, IMG_SIZE, IMG_SIZE)), torch.zeros((SEQ_LEN, AUD_BINS))
    for t in range(SEQ_LEN):
        x = t % IMG_SIZE
        v[t, 4:8, x:min(x+3, IMG_SIZE)] = 1.0
        freq = 6 + int((x / IMG_SIZE) * 4)
        a[t, freq:freq+2] = 1.0 
    return v.view(SEQ_LEN, -1), a

def generate_ambulance():
    v, a = torch.zeros((SEQ_LEN, IMG_SIZE, IMG_SIZE)), torch.zeros((SEQ_LEN, AUD_BINS))
    for t in range(SEQ_LEN):
        if t % 4 < 2:
            v[t, 4:8, 4:8] = 1.0 
            a[t, 9:11] = 1.0     
        else:
            a[t, 5:7] = 1.0      
    return v.view(SEQ_LEN, -1), a

def generate_rain():
    v, a = torch.zeros((SEQ_LEN, IMG_SIZE, IMG_SIZE)), torch.zeros((SEQ_LEN, AUD_BINS))
    for t in range(SEQ_LEN):
        v[t] = (torch.rand((IMG_SIZE, IMG_SIZE)) > 0.85).float()
        a[t, :] = torch.rand(AUD_BINS) * 0.5 + 0.2
    return v.view(SEQ_LEN, -1), a

def generate_balloon():
    v, a = torch.zeros((SEQ_LEN, IMG_SIZE, IMG_SIZE)), torch.zeros((SEQ_LEN, AUD_BINS))
    for t in range(SEQ_LEN):
        y = IMG_SIZE - 1 - t  
        if y >= 0:
            v[t, y:y+2, 5:7] = 1.0
            a[t, min(t, AUD_BINS-1)] = 1.0 
    return v.view(SEQ_LEN, -1), a

def generate_metronome():
    v, a = torch.zeros((SEQ_LEN, IMG_SIZE, IMG_SIZE)), torch.zeros((SEQ_LEN, AUD_BINS))
    for t in range(SEQ_LEN):
        x = 2 if t % 4 < 2 else 9 
        v[t, 8:10, x:x+2] = 1.0
        if t % 4 == 0 or t % 4 == 2:
            a[t, 5] = 1.0 
    return v.view(SEQ_LEN, -1), a

EXPERIENCES = {
    '1': ('Ball', generate_bouncing_ball()), '2': ('Train', generate_passing_train()),
    '3': ('Ambulance', generate_ambulance()), '4': ('Rain', generate_rain()),
    '5': ('Balloon', generate_balloon()), '6': ('Metronome', generate_metronome())
}

# --- 2. THE ORGANIC RESONANT LATENT SPACE ---
class OrganicResonantMedium(nn.Module):
    def __init__(self):
        super().__init__()
        positions = []
        for z in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    px = x + (y % 2) * 0.5 + (z % 2) * 0.5
                    py = y * np.sqrt(3)/2 + (z % 2) * np.sqrt(3)/6
                    pz = z * np.sqrt(2/3)
                    positions.append([px * SPACING, py * SPACING, pz * SPACING])
                    
        self.N = len(positions)
        self.pos = torch.tensor(positions, dtype=torch.float32).to(DEVICE)
        
        self.mask = torch.zeros((self.N, self.N)).to(DEVICE)
        for i in range(self.N):
            for j in range(self.N):
                if torch.norm(self.pos[i] - self.pos[j]) < SPACING * 1.5:
                    self.mask[i, j] = 1.0
                    
        self.raw_tension = nn.Parameter(torch.randn(self.N, self.N) * 0.05)
        self.omega = nn.Parameter(torch.rand(self.N) * 2.0 + 1.0)
        
        self.retina_in = nn.Linear(IMG_SIZE * IMG_SIZE, self.N)
        self.cochlea_in = nn.Linear(AUD_BINS, self.N)
        self.visual_cortex_out = nn.Linear(self.N, IMG_SIZE * IMG_SIZE)
        self.auditory_cortex_out = nn.Linear(self.N, AUD_BINS)

    def get_tension_matrix(self):
        W = torch.abs(self.raw_tension)
        return ((W + W.T) / 2.0) * self.mask

    # MODIFIED: Accepts U_in and V_in so momentum can carry over sequences!
    def forward(self, video_seq, audio_seq, U_in=None, V_in=None):
        U = torch.zeros(self.N).to(DEVICE) if U_in is None else U_in
        V = torch.zeros(self.N).to(DEVICE) if V_in is None else V_in
        Tension = self.get_tension_matrix()
        
        out_video, out_audio, energy_history = [], [], []
        
        for t in range(SEQ_LEN):
            sensory_force = torch.zeros(self.N).to(DEVICE)
            if torch.sum(video_seq[t]) > 0: sensory_force += self.retina_in(video_seq[t])
            if torch.sum(audio_seq[t]) > 0: sensory_force += self.cochlea_in(audio_seq[t])
            
            pull_from_neighbors = torch.matmul(U, Tension)
            anchor_drag = U * torch.sum(Tension, dim=0)
            spring_forces = pull_from_neighbors - anchor_drag
            
            restoring_forces = -(self.omega ** 2) * U
            damping_forces = -0.1 * V
            
            acceleration = spring_forces + restoring_forces + damping_forces + sensory_force
            V = V + acceleration * DT
            U = torch.tanh(U + V * DT) 
            
            v_frame = torch.sigmoid(self.visual_cortex_out(U))
            a_frame = torch.sigmoid(self.auditory_cortex_out(U))
            
            out_video.append(v_frame)
            out_audio.append(a_frame)
            energy_history.append(torch.abs(U).clone())
            
        return torch.stack(out_video), torch.stack(out_audio), torch.stack(energy_history), Tension, U, V

# --- 3. THE LAB UI ---
class ToddlerLab:
    def __init__(self):
        print(f"Initializing Brain on {DEVICE}...")
        self.model = OrganicResonantMedium().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.mse = nn.MSELoss()
        
        self.mode = "IDLE"
        self.current_exp_key = '1'
        
        # DREAM STATE VARIABLES
        self.dream_U = None
        self.dream_V = None
        self.dream_v_in = None
        self.dream_a_in = None
        
        self.setup_ui()
        
    def setup_ui(self):
        plt.ion()
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.canvas.manager.set_window_title('Toddler Brain V3: Autoregressive Dreaming')
        gs = GridSpec(2, 4, figure=self.fig, height_ratios=[1, 1.5])
        
        self.ax_v_true = self.fig.add_subplot(gs[0, 0]); self.ax_v_true.set_title("True / Dream Input")
        self.ax_a_true = self.fig.add_subplot(gs[0, 1]); self.ax_a_true.set_title("True / Dream Input")
        self.ax_v_pred = self.fig.add_subplot(gs[0, 2]); self.ax_v_pred.set_title("Predicted / Dream Output")
        self.ax_a_pred = self.fig.add_subplot(gs[0, 3]); self.ax_a_pred.set_title("Predicted / Dream Output")
        self.ax_3d = self.fig.add_subplot(gs[1, :], projection='3d'); self.ax_3d.set_title("The Resonant Brain")
        
        for ax in [self.ax_v_true, self.ax_v_pred]: ax.axis('off')
            
        self.img_v_true = self.ax_v_true.imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap='gray', vmin=0, vmax=1)
        self.img_v_pred = self.ax_v_pred.imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap='plasma', vmin=0, vmax=1)
        
        self.bar_a_true = self.ax_a_true.bar(range(AUD_BINS), np.zeros(AUD_BINS), color='cyan')
        self.bar_a_pred = self.ax_a_pred.bar(range(AUD_BINS), np.zeros(AUD_BINS), color='magenta')
        self.ax_a_true.set_ylim(0, 1.2); self.ax_a_pred.set_ylim(0, 1.2)
        
        self.pos = self.model.pos.cpu().numpy()
        self.scatter = self.ax_3d.scatter(self.pos[:,0], self.pos[:,1], self.pos[:,2], s=50, c='gray')
        
        self.ax_3d.set_facecolor('black'); self.fig.patch.set_facecolor('black')
        self.ax_3d.axis('off')
        
        for text in [self.ax_v_true, self.ax_a_true, self.ax_v_pred, self.ax_a_pred, self.ax_3d]:
            text.title.set_color('white'); text.tick_params(colors='white')
            
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key in EXPERIENCES.keys():
            self.current_exp_key = event.key
            self.mode = f"TRAIN: {EXPERIENCES[self.current_exp_key][0]}"
        elif event.key == 'q':
            self.mode = f"IMAGINE: Hear Audio -> Predict Vision ({EXPERIENCES[self.current_exp_key][0]})"
        elif event.key == 'w':
            self.mode = f"IMAGINE: See Vision -> Predict Audio ({EXPERIENCES[self.current_exp_key][0]})"
        elif event.key == 'd':
            self.mode = "DREAMING"
            # Kickstart the dream with random neural static
            self.dream_v_in = (torch.rand((SEQ_LEN, IMG_SIZE*IMG_SIZE)) > 0.95).float().to(DEVICE)
            self.dream_a_in = (torch.rand((SEQ_LEN, AUD_BINS)) > 0.95).float().to(DEVICE)
            self.dream_U = torch.randn(self.model.N).to(DEVICE) * 0.5
            self.dream_V = torch.randn(self.model.N).to(DEVICE) * 0.5
        elif event.key == '9':
            torch.save(self.model.state_dict(), SAVE_PATH); print(f"\n💾 Memory Saved!")
        elif event.key == '0':
            if os.path.exists(SAVE_PATH): self.model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE)); print(f"\n📂 Memory Loaded!")
        elif event.key == ' ':
            self.mode = "IDLE"
            
        print(f"🧠 Mode: {self.mode}")

    def run(self):
        print("\n=== TODDLER BRAIN LAB V3 ===")
        print("EXPERIENCES: [1] Ball, [2] Train, [3] Ambulance, [4] Rain, [5] Balloon, [6] Metronome")
        print("Hold Number Key to TRAIN.")
        print("Press [q] or [w] to IMAGINE missing senses.")
        print("Press [d] to induce a continuous DREAM STATE.")
        print("[9] Save  |  [0] Load  |  [Space] Pause")
        
        while True:
            if not plt.fignum_exists(self.fig.number): break
            if self.mode == "IDLE": 
                plt.pause(0.1); continue
                
            current_mode = self.mode 
            _, (v_true, a_true) = EXPERIENCES[self.current_exp_key]
            v_true, a_true = v_true.to(DEVICE), a_true.to(DEVICE)
            
            if "TRAIN" in current_mode:
                v_in, a_in = v_true, a_true
            elif "Hear Audio" in current_mode:
                v_in, a_in = torch.zeros_like(v_true), a_true
            elif "See Vision" in current_mode:
                v_in, a_in = v_true, torch.zeros_like(a_true)
            elif current_mode == "DREAMING":
                v_in, a_in = self.dream_v_in, self.dream_a_in
            else: continue
                
            if "TRAIN" in current_mode:
                self.optimizer.zero_grad()
                v_pred, a_pred, energy, tension, _, _ = self.model(v_in, a_in)
                loss = self.mse(v_pred, v_true) + self.mse(a_pred, a_true) + 0.05 * torch.mean(torch.abs(tension))
                loss.backward()
                self.optimizer.step()
            elif current_mode == "DREAMING":
                with torch.no_grad():
                    # Pass the previous momentum into the network
                    v_pred, a_pred, energy, tension, self.dream_U, self.dream_V = self.model(
                        v_in, a_in, U_in=self.dream_U, V_in=self.dream_V
                    )
                    
                    # AUTOREGRESSIVE FEEDBACK
                    # We inject 1% random noise (spontaneous neuron firing) to keep the dream from fading
                    noise_v = (torch.rand_like(v_pred) > 0.99).float()
                    noise_a = (torch.rand_like(a_pred) > 0.99).float()
                    
                    # Threshold predictions so the brain commits to the hallucination (keeps images sharp)
                    self.dream_v_in = torch.clamp((v_pred.detach() > 0.35).float() + noise_v, 0, 1)
                    self.dream_a_in = torch.clamp((a_pred.detach() > 0.35).float() + noise_a, 0, 1)
            else:
                with torch.no_grad():
                    v_pred, a_pred, energy, tension, _, _ = self.model(v_in, a_in)
                    
            tens_np = tension.detach().cpu().numpy()
            
            for t in range(SEQ_LEN):
                try:
                    self.img_v_true.set_data(v_in[t].view(IMG_SIZE, IMG_SIZE).detach().cpu().numpy())
                    self.img_v_pred.set_data(v_pred[t].view(IMG_SIZE, IMG_SIZE).detach().cpu().numpy())
                    
                    a_in_np, a_pred_np = a_in[t].detach().cpu().numpy(), a_pred[t].detach().cpu().numpy()
                    for i, rect in enumerate(self.bar_a_true): rect.set_height(float(a_in_np[i]))
                    for i, rect in enumerate(self.bar_a_pred): rect.set_height(float(a_pred_np[i]))
                    
                    eng_np = energy[t].detach().cpu().numpy()
                    colors = np.zeros((self.model.N, 4))
                    for i in range(self.model.N):
                        c_val = min(float(eng_np[i]) * 1.5, 1.0)
                        colors[i] = [c_val, c_val*0.3, 1.0, 0.2 + c_val * 0.8] 
                        
                    self.scatter.set_color(colors)
                    self.scatter.set_sizes(10 + eng_np * 250)
                    
                    if t == 0: 
                        while len(self.ax_3d.lines) > 0: self.ax_3d.lines[-1].remove()
                        for i in range(self.model.N):
                            for j in range(i+1, self.model.N):
                                w = float(tens_np[i, j])
                                if w > 0.08:
                                    self.ax_3d.plot([self.pos[i,0], self.pos[j,0]], 
                                                    [self.pos[i,1], self.pos[j,1]], 
                                                    [self.pos[i,2], self.pos[j,2]], 
                                                    color='cyan', alpha=min(w*2, 1.0), linewidth=w*5)
                                                    
                    self.ax_3d.set_title(f"Internal Brain Wave (Frame {t+1}/{SEQ_LEN}) | {current_mode}")
                    plt.pause(0.015) 
                
                except Exception as e: break

if __name__ == "__main__":
    lab = ToddlerLab()
    lab.run()