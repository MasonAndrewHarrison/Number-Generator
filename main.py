import torch
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from model import Discriminator, Generator

device = "cuda" if torch.cuda.is_available() else "cpu"
gen_model = Generator(100, 1, 64).to(device)
gen_model.load_state_dict(torch.load("Generator_Weights.pth", map_location=device))
gen_model.eval()


z_noise = torch.randn(1, 100, 1, 1).to(device)
image = gen_model(z_noise).view(64, 64).detach().cpu()


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

img_display = ax.imshow(image, cmap="gray")

ax_z = plt.axes([0.25, 0.1, 0.65, 0.03])
slider_z = Slider(ax_z, 'z[0]', valmin=-5.0, valmax=5.0, valinit=z_noise[0, 0, 0, 0].item())

ax_z2 = plt.axes([0.25, 0.0, 0.65, 0.03])
slider_z2 = Slider(ax_z2, 'z[0]', valmin=-5.0, valmax=5.0, valinit=z_noise[0, 1, 0, 0].item())

def update(val, index):
    z_noise[0, index, 0, 0] = torch.tensor(val, device=device)
    with torch.no_grad():
        new_image = gen_model(z_noise).view(64, 64).detach().cpu()
    img_display.set_data(new_image)
    fig.canvas.draw_idle()

slider_z.on_changed(lambda x: update(x, 0))
slider_z2.on_changed(lambda x: update(x, 1))

plt.show()