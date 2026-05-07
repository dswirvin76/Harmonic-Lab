import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import collections
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAZE_SIZE = 35          
SEQ_LEN = 12            
LATENT_DIM = 128
WEIGHTS_FILE = "harmonic_antenna2.pth"
CONFIDENCE_THRESHOLD = 0.85  # Used for visuals only now (Cyan vs Red filament)

MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)] 
WAIT_IDX = 4

# --- 1. Environment ---
class MazeEnv:
    def __init__(self, size):
        self.size = size
        self.reset()

    def reset(self):
        self.grid = np.ones((self.size, self.size), dtype=np.float32) 
        self.visited = np.zeros((self.size, self.size), dtype=bool)
        self.start = (1, 1)
        self.goal = (self.size-2, self.size-2)
        self.current_pos = list(self.start)
        self.generate_maze(1, 1)
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        self.trace_layer = np.zeros_like(self.grid)
        self.trace_layer[tuple(self.start)] = 1.0
        self.imagination_layer = np.zeros_like(self.grid)
        return self.get_observation()

    def generate_maze(self, r, c):
        self.visited[r, c] = True
        self.grid[r, c] = 0
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 1 <= nr < self.size-1 and 1 <= nc < self.size-1 and not self.visited[nr, nc]:
                self.grid[r + dr//2, c + dc//2] = 0 
                self.generate_maze(nr, nc)

    def get_observation(self):
        obs = np.zeros((4, self.size, self.size), dtype=np.float32)
        obs[0] = self.grid
        obs[1] = self.trace_layer
        obs[2, self.current_pos[0], self.current_pos[1]] = 1.0
        obs[2, self.goal[0], self.goal[1]] = 0.5
        obs[3] = self.imagination_layer 
        return torch.from_numpy(obs).unsqueeze(0).to(DEVICE)

    def draw_imagination(self, move_indices):
        self.imagination_layer.fill(0)
        tr, tc = self.current_pos
        for m in move_indices:
            if m == WAIT_IDX: break 
            dr, dc = MOVES[m]
            tr, tc = tr + dr, tc + dc
            if 0 <= tr < self.size and 0 <= tc < self.size:
                self.imagination_layer[tr, tc] = 1.0
            else: break

    def get_optimal_path(self):
        q = collections.deque([(self.current_pos[0], self.current_pos[1], [])])
        visited = set()
        visited.add(tuple(self.current_pos))
        while q:
            r, c, path = q.popleft()
            if (r, c) == self.goal: return path 
            for i, (dr, dc) in enumerate(MOVES[:4]): 
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.grid[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc, path + [i]))
        return []

    def step(self, move_idx):
        if move_idx == WAIT_IDX: return False, False
        dr, dc = MOVES[move_idx]
        nr, nc = self.current_pos[0] + dr, self.current_pos[1] + dc
        if self.grid[nr, nc] == 1: return False, False # Wall
        
        self.current_pos = [nr, nc]
        self.trace_layer[nr, nc] = 1.0 
        done = (tuple(self.current_pos) == self.goal)
        return True, done

# --- 2. Harmonic Planner ---
class HarmonicCell(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.base_freqs = nn.Parameter(torch.linspace(0.1, 15.0, hidden_dim), requires_grad=False)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.damping = nn.Parameter(torch.tensor(0.98))

    def step(self, h_prev, t, freq_mod, amp_mod):
        W_rec = self.A - self.A.T
        osc = amp_mod * torch.sin((self.base_freqs + freq_mod) * t)
        h_next = (h_prev * self.damping) + 0.1 * (h_prev @ W_rec + osc)
        return torch.tanh(h_next)

# --- 2. Harmonic Planner v2 (Phase & Overtones) ---
class HarmonicCell_v2(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # We initialize base frequencies logarithmically to cover short & long term plans
        self.base_freqs = nn.Parameter(torch.logspace(-1, 1, hidden_dim), requires_grad=False)
        
        # Skew-Symmetric Matrix for rotational dynamics
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        
        # Learnable damping per frequency (some thoughts fade fast, others stay)
        self.damping = nn.Parameter(torch.ones(hidden_dim) * 0.98)

    def step(self, h_prev, t, freq_mod, amp_mod, phase_mod):
        # 1. Rotational Coupling
        W_rec = self.A - self.A.T
        
        # 2. The Wave Equation with PHASE
        # sin( omega * t + phi )
        total_freq = self.base_freqs + freq_mod
        wave = torch.sin((total_freq * t) + phase_mod)
        
        # 3. Overtones (Sharpness)
        # We add a 2nd harmonic (2x freq) to allow for sharper turns
        overtone = 0.5 * torch.sin((total_freq * 2.0 * t) + phase_mod)
        
        # 4. Energy Injection
        osc = amp_mod * (wave + overtone)
        
        # 5. Integration
        # Damping ensures the system is stable, W_rec rotates it, osc drives it.
        h_next = (h_prev * self.damping) + 0.1 * (h_prev @ W_rec + osc)
        
        return torch.tanh(h_next)

class HarmonicPlanner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        flat_size = 64 * MAZE_SIZE * MAZE_SIZE
        
        self.fc_init = nn.Linear(flat_size, LATENT_DIM)
        
        # Modulators
        self.fc_freq  = nn.Linear(flat_size, LATENT_DIM) # "What" to do
        self.fc_amp   = nn.Linear(flat_size, LATENT_DIM) # "How strongly" to do it
        self.fc_phase = nn.Linear(flat_size, LATENT_DIM) # "When" to do it (NEW)

        self.cell = HarmonicCell_v2(LATENT_DIM)
        self.action_head = nn.Linear(LATENT_DIM, 5)

    def forward(self, x, h_real_prev=None):
        feats = self.encoder(x)
        visual_h = self.fc_init(feats)
        
        # Continuous Memory Integration
        if h_real_prev is not None:
            h = torch.tanh(visual_h + 0.5 * h_real_prev)
        else:
            h = torch.tanh(visual_h)
            
        h_real_next = h.clone().detach()

        # Calculate Harmonic Parameters
        freq = torch.tanh(self.fc_freq(feats))   # Frequency Shift
        amp = torch.sigmoid(self.fc_amp(feats))  # Amplitude
        phase = torch.tanh(self.fc_phase(feats)) * 3.14159 # Phase Shift (-pi to pi)
        
        logits_seq = []
        for t in range(SEQ_LEN):
            # Pass phase into the cell
            h = self.cell.step(h, t * 0.2, freq, amp, phase)
            logits_seq.append(self.action_head(h))
            
        return torch.stack(logits_seq, dim=1), h_real_next

# --- 3. Interactive Lab ---
class MazeLab:
    def __init__(self, model):
        self.model = model
        self.env = MazeEnv(MAZE_SIZE)
        self.is_training = True
        
        plt.ion()
        gs = dict(width_ratios=[3, 1])
        self.fig, (self.ax_maze, self.ax_stats) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw=gs)
        self.fig.canvas.manager.set_window_title('Harmonic Antenna - Emergent Entity')
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.00003) #was 0.0001 during training
        self.criterion = nn.CrossEntropyLoss()
        self.loss_history = []
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.loop()

    def on_key(self, event):
        if event.key == 'd':
            self.is_training = not self.is_training
            print(f"--- Mode: {'TRAINING' if self.is_training else 'DREAMING'} ---")
        elif event.key == 's':
            torch.save(self.model.state_dict(), WEIGHTS_FILE)
            print("Brain Saved.")

    def loop(self):
        epoch = 0
        while True:
            obs = self.env.reset()
            done = False
            total_moves = 0
            time_steps = 0
            
            # The creature's continuous memory state resets per maze
            h_real = None 
            
            # Increased tick limit so the 'fly' has time to wander and correct itself
            while not done and time_steps < 900:
                if not plt.fignum_exists(self.fig.number): return 

                full_path = self.env.get_optimal_path()
                
                # --- A. BRAIN PROCESSING (Forward Pass) ---
                if self.is_training:
                    self.model.train()
                    target_seq = full_path[:SEQ_LEN]
                    pad_len = SEQ_LEN - len(target_seq)
                    target_seq = target_seq + [WAIT_IDX] * pad_len 
                    target_tensor = torch.tensor([target_seq], device=DEVICE)

                    self.optimizer.zero_grad()
                    # Pass previous physical memory into the current thought
                    logits, h_real = self.model(obs, h_real) 
                    
                    # Discounted Loss: Cares most about the immediate next step
                    loss = 0
                    discount = 1.0
                    for i in range(SEQ_LEN):
                        loss += self.criterion(logits[:, i, :], target_tensor[:, i]) * discount
                        discount *= 0.85 
                        
                    loss.backward()
                    self.optimizer.step()
                    self.loss_history.append(loss.item())

                else:
                    self.model.eval()
                    with torch.no_grad():
                        logits, h_real = self.model(obs, h_real)

                # --- B. THE FLY IN A JAR (Stochastic Sampling) ---
                probs = F.softmax(logits, dim=2)
                first_step_probs = probs[0, 0] # Probabilities for immediate next move
                
                # Sample based on probability urges rather than strictly gating by a threshold
                dist = torch.distributions.Categorical(first_step_probs)
                first_move = dist.sample().item()
                first_move_conf = first_step_probs[first_move].item()
                
                # It acts on its sample, unless the network explicitly chose WAIT
                should_move = (first_move != WAIT_IDX)
                
                # For Visualizing the Antenna, we still show its absolute "best guess" (argmax)
                _, pred_indices = torch.max(probs, dim=2)
                pred_seq = pred_indices.cpu().numpy()[0]
                self.env.draw_imagination(pred_seq)

                # --- C. EXECUTE MOVEMENT ---
                moved = False
                if should_move:
                    moved, done = self.env.step(first_move)
                    if moved: total_moves += 1

                # ---> BUG FIX: UPDATE THE EYES <---
                # The agent must see that it has moved to a new tile
                obs = self.env.get_observation()

                # Render
                self.render(pred_seq, first_move_conf, moved, epoch, time_steps)
                plt.pause(0.01) 
                time_steps += 1
            
            if self.is_training:
                if epoch % 5 == 0: torch.save(self.model.state_dict(), WEIGHTS_FILE)
            epoch += 1

    def render(self, projected_seq, confidence, moved, epoch, time_steps):
        self.ax_maze.clear()
        
        grid = self.env.grid
        trace = self.env.trace_layer
        
        img = np.ones((MAZE_SIZE, MAZE_SIZE, 3))
        img[grid==1] = 0 
        img[trace==1] = [0.6, 1.0, 0.6] 
        
        self.ax_maze.imshow(img, interpolation='nearest')
        r, c = self.env.current_pos
        
        agent_col = 'red' if moved else 'white'
        self.ax_maze.plot(c, r, 'o', color=agent_col, markersize=10, markeredgecolor='black') 
        gr, gc = self.env.goal
        self.ax_maze.plot(gc, gr, 'bx', markersize=12, markeredgewidth=3)
        
        # Draw Antenna Filament
        curr_r, curr_c = r, c
        for i, m in enumerate(projected_seq):
            if m == WAIT_IDX: break
            dr, dc = MOVES[m]
            next_r, next_c = curr_r + dr, curr_c + dc
            
            col = 'cyan' if confidence > CONFIDENCE_THRESHOLD else 'red'
            
            self.ax_maze.plot([curr_c, next_c], [curr_r, next_r], 
                              color=col, linestyle='--', alpha=0.6, linewidth=2)
            curr_r, curr_c = next_r, next_c
            
        mode_str = "TRAINING" if self.is_training else "DREAMING"
        self.ax_maze.set_title(f"{mode_str} | Ep {epoch} | T: {time_steps}")
        
        # Stats Bar
        self.ax_stats.clear()
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.axis('off')
        
        self.ax_stats.text(0.5, 0.1, "Sampled Confidence", ha='center')
        bar_col = 'lime' if confidence > CONFIDENCE_THRESHOLD else 'orange'
        self.ax_stats.bar([0.5], [confidence], width=0.5, color=bar_col)
        self.ax_stats.axhline(CONFIDENCE_THRESHOLD, color='red', linestyle=':', label='Threshold')
        self.ax_stats.text(0.5, confidence + 0.02, f"{confidence:.2f}", ha='center')
        
        self.fig.canvas.draw()

if __name__ == "__main__":
    model = HarmonicPlanner().to(DEVICE)
    if os.path.exists(WEIGHTS_FILE):
        print(f"Loading {WEIGHTS_FILE}...")
        try:
            model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
        except:
            print("Fresh start.")
    MazeLab(model)