import torch
from torch.utils.data import DataLoader
from model_unet import ConditionalUNet
from diffusion import Diffusion
from dataset import CT_Dataset
import torch.nn.functional as F
import os
import glob

if torch.cuda.is_available():
    device = "cuda"
    print("cuda is available")
else:
    device = "cpu"
    print("cuda is not available")

# dataset paths
low_paths = sorted(glob.glob(r"G:\for_use\dataset\2\Preprocessed_512x512\512\Full Dose\1mm\Sharp Kernel (D45)\L067*.png"))   # 或 .jpg / .tif / .npy
high_paths = sorted(glob.glob(r"G:\for_use\dataset\2\Preprocessed_512x512\512\Quarter Dose\1mm\Sharp Kernel (D45)\L067*.png"))

print("low ct 数量:", len(low_paths))
print("high ct 数量:", len(high_paths))

dataset = CT_Dataset(low_paths, high_paths)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# models
model = ConditionalUNet().to(device)
diff = Diffusion(timesteps=1000)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 200

for epoch in range(epochs):
    for low, high in loader:
        low, high = low.to(device), high.to(device)

        t = torch.randint(0, diff.timesteps, (low.size(0),), device=device)
        noise = torch.randn_like(high)

        x_t = diff.q_sample(high, t, noise)

        pred_noise = model(x_t, t.float(), cond=low)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} | loss={loss.item():.4f}")

torch.save(model.state_dict(), "ct_diffusion.pth")
