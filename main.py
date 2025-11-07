import torch
import os
import matplotlib.pyplot as plt
from model import Discriminator, Generator

device = "cuda" if torch.cuda.is_available() else "cpu"
gen_model = Generator(100, 1, 64).to(device)
gen_model.load_state_dict(torch.load("Generator_Weights.pth", map_location=device))
gen_model.eval()


z_noise = torch.randn(1, 100, 1, 1).to(device)
image = gen_model(z_noise).view(64, 64).detach().cpu()


plt.imshow(image, cmap="gray")
plt.show()


