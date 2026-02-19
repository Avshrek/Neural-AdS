import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os

# --- CONFIGURATION ---
MODEL_PATH = os.path.join("models", "final_fno_model.pth") 
GRID_SIZE = 64
MODES = 12
WIDTH = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. MODEL ARCHITECTURE (Standard FNO)
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(2, self.width) 
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1); self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1); self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128); self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x).permute(0, 3, 1, 2)
        x1 = self.conv0(x); x2 = self.w0(x); x = F.gelu(x1 + x2)
        x1 = self.conv1(x); x2 = self.w1(x); x = F.gelu(x1 + x2)
        x1 = self.conv2(x); x2 = self.w2(x); x = F.gelu(x1 + x2)
        x1 = self.conv3(x); x2 = self.w3(x); x = x1 + x2
        x = x.permute(0, 2, 3, 1)
        x = self.fc2(F.gelu(self.fc1(x)))
        return x.squeeze()

# ==========================================
# 2. GENERATE RANDOM TEST UNIVERSE
# ==========================================
def get_test_sample(grid_size=64):
    N = grid_size * grid_size
    main_diag = -4 * np.ones(N)
    side_diag = np.ones(N - 1); side_diag[np.arange(1, N) % grid_size == 0] = 0
    up_down_diag = np.ones(N - grid_size)
    A = sp.diags([main_diag, side_diag, side_diag, up_down_diag, up_down_diag], [0, -1, 1, -grid_size, grid_size], format='csc')
    grid_idx = np.arange(N).reshape(grid_size, grid_size)
    boundaries = np.unique(np.concatenate((grid_idx[0,:], grid_idx[-1,:], grid_idx[:,0], grid_idx[:,-1])))
    for idx in boundaries:
        A[idx, :] = 0; A[idx, idx] = 1
    solve = spla.factorized(A)

    x_grid = np.linspace(0, 2*np.pi, grid_size)
    k1, k2 = np.random.randint(1, 4), np.random.randint(1, 4)
    p1, p2 = np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)
    boundary = np.sin(k1*x_grid + p1) + 0.5 * np.cos(k2*x_grid + p2)
    
    rhs = np.zeros(N); rhs[grid_idx[0,:]] = boundary
    sol = solve(rhs).reshape(grid_size, grid_size)
    
    x_input = np.zeros((grid_size, grid_size, 2))
    x_input[:, :, 0] = np.tile(boundary, (grid_size, 1))
    x_input[:, :, 1] = np.tile(np.linspace(0, 1, grid_size).reshape(-1, 1), (1, grid_size))
    
    return x_input, sol

# ==========================================
# 3. EVALUATION
# ==========================================
model = FNO2d(MODES, MODES, WIDTH).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

x_raw, y_raw = get_test_sample(GRID_SIZE)
x_norm = (x_raw - x_raw.mean()) / (x_raw.std() + 1e-8)
x_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).to(device)

with torch.no_grad():
    prediction_norm = model(x_tensor).cpu().numpy()

prediction_physical = prediction_norm * y_raw.std() + y_raw.mean()

# --- METRIC CALCULATION ---
# Global MAE (The Physics Engine Score)
mae = np.mean(np.abs(y_raw - prediction_physical))

# Max Error Percentage (The Interior Stress Test)
valid_y = y_raw[2:, :]
valid_pred = prediction_physical[2:, :]
max_err = np.max(np.abs(valid_y - valid_pred))
data_range = np.max(y_raw) - np.min(y_raw)
max_err_pct = (max_err / data_range) * 100

# Visualization
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
im1 = ax[0].imshow(y_raw, cmap='inferno', vmin=-1, vmax=1)
ax[0].set_title("Ground Truth (Physics)")
plt.colorbar(im1, ax=ax[0])

im2 = ax[1].imshow(prediction_physical, cmap='inferno', vmin=-1, vmax=1)
ax[1].set_title(f"Neural-AdS AI (MAE: {mae:.5f})")
plt.colorbar(im2, ax=ax[1])

im3 = ax[2].imshow(np.abs(y_raw - prediction_physical), cmap='binary', vmin=0, vmax=max_err) 
ax[2].set_title(f"Error (Max Interior: {max_err_pct:.2f}%)")
plt.colorbar(im3, ax=ax[2])

plt.savefig("holographic_impact_final.png")
plt.show()
