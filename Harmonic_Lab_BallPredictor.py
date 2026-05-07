import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import os

# --- Hyperparameters ---
FRAME_SIZE = 56
HIDDEN_DIM = 512
LEARNING_RATE = 0.003 # Lowered for stability
DT = 0.1 
BALL_RADIUS = 3
WEIGHTS_PATH = "harmonic_model_v1.pth"

class HarmonicPredictiveCoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Mapping inputs to the "strings"
        self.W_in = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        # Skew-Symmetric kernel kernel
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        # Frequencies
        self.freqs = nn.Parameter(torch.linspace(0.1, 2.0, hidden_dim))
        # Output mapping
        self.W_out = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        
    def get_skew_rec(self):
        return self.A - self.A.T

    def step(self, x_current, h_prev, t):
        W_rec = self.get_skew_rec()
        
        # 1. Harmonic Oscillation (internal rhythm)
        oscillation = torch.sin(self.freqs * t)
        
        # 2. Resonant Interaction
        interaction = h_prev @ W_rec.T
        
        # 3. Input Pluck
        pluck = x_current @ self.W_in.T
        
        # 4. State Update (with damping/tanh to prevent explosion)
        h_next = h_prev + DT * (interaction + oscillation + pluck)
        h_next = torch.tanh(h_next) 
        
        return h_next

    def train_resonant(self, steps, simulator):
        self.train()
        print("--- Tuning the Harmonic Neurons ---")
        h = torch.zeros(1, self.hidden_dim)
        
        for i in range(steps):
            t = i * DT
            # Get data and convert to tensor
            curr_np = simulator.draw()
            curr_frame = torch.from_numpy(curr_np).float().flatten().unsqueeze(0)
            
            simulator.step()
            
            next_np = simulator.draw()
            next_frame = torch.from_numpy(next_np).float().flatten().unsqueeze(0)
            
            # Predict
            h_next = self.step(curr_frame, h, t)
            pred_next = torch.sigmoid(h_next @ self.W_out.T)
            
            # Error (Predictive Coding)
            error = next_frame - pred_next
            h_error = error @ self.W_out
            
            # Manual Gradients with Gradient Clipping for stability
            with torch.no_grad():
                grad_out = (error.T @ h_next).clamp(-1, 1)
                grad_in  = (h_error.T @ curr_frame).clamp(-1, 1)
                grad_A   = (0.5 * h_error.T @ h).clamp(-1, 1)
                
                self.W_out += LEARNING_RATE * grad_out
                self.W_in  += LEARNING_RATE * grad_in
                self.A     += LEARNING_RATE * grad_A
                
            h = h_next.detach()
            if i % 500 == 0:
                loss = error.pow(2).mean().item()
                print(f"Step {i} | Loss: {loss:.6f}")

    def dream(self, seed_frame_np):
        self.eval()
        # FIX: Convert numpy seed to torch tensor
        seed_tensor = torch.from_numpy(seed_frame_np).float().flatten().unsqueeze(0)
        h = torch.tanh(seed_tensor @ self.W_in.T)
        
        fig, ax = plt.subplots()
        img = ax.imshow(seed_frame_np, cmap='magma', vmin=0, vmax=1)
        ax.set_title("Harmonic Resonance Dream (Vibrating Memory)")
        
        def animate(i):
            nonlocal h
            t = i * DT
            with torch.no_grad():
                # Self-perpetuating cycle
                temp = 0.5
                current_view = torch.sigmoid(h @ self.W_out.T / temp)
                h = self.step(current_view, h, t)
                
                frame = torch.sigmoid(h @ self.W_out.T).reshape(FRAME_SIZE, FRAME_SIZE).numpy()
                img.set_array(frame)
            return [img]

        ani = animation.FuncAnimation(fig, animate, frames=1000, interval=30, blit=True)
        plt.show()

# --- Re-import Simulator or paste here ---
from pong_predictor2 import BouncingBallSimulator

if __name__ == "__main__":
    model = HarmonicPredictiveCoder(FRAME_SIZE**2, HIDDEN_DIM)

    # --- LOAD WEIGHTS IF THEY EXIST ---
    if os.path.exists(WEIGHTS_PATH):
        print(f"--- Loading existing weights from {WEIGHTS_PATH} ---")
        model.load_state_dict(torch.load(WEIGHTS_PATH))
    else:
        print("--- No saved weights found. Starting from scratch. ---")

    sim = BouncingBallSimulator(FRAME_SIZE, radius=BALL_RADIUS)
    
    # 1. Train
    model.train_resonant(3000, sim)
    # Save weights after training
    torch.save(model.state_dict(), "harmonic_model_v1.pth")
    print("Model weights saved to harmonic_model_v1.pth")
    
    # 2. Dream
    seed = sim.draw()
    model.dream(seed)