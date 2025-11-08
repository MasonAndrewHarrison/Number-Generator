import torch
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from model import Discriminator, Generator

device = "cuda" if torch.cuda.is_available() else "cpu"
gen_model = Generator(100, 1, 64).to(device)
gen_model.load_state_dict(torch.load("Generator_Weights.pth", map_location=device))
gen_model.eval()


z_noise = torch.randn(1, 100, 1, 1).to(device)
image = gen_model(z_noise).view(64, 64).detach().cpu()

fig, ax = plt.subplots(figsize=(21, 15))
plt.subplots_adjust(bottom=0.25, left=0.6)

img_display = ax.imshow(image, cmap="gray")
slider_list = [0]*100

for i in range(10):
    for j in range(10):

        true_index = j+(i*10)
        print(true_index)
        ax_z = plt.axes([0.03 + (j/20), 0.9 - (i/10.5), 0.03, 0.067])
        slider_list[true_index] = Slider(ax_z, true_index+1, valmin=-4.0, valmax=4.0, valinit=z_noise[0, true_index, 0, 0].item(), orientation="vertical")

button_ax = plt.axes([.65, .25, .2, .05])
button = Button(
    button_ax,
    "Reset Z Noise",
)

def update(val, index):
    z_noise[0, index, 0, 0] = torch.tensor(val, device=device)
    with torch.no_grad():
        new_image = gen_model(z_noise).view(64, 64).detach().cpu()
    img_display.set_data(new_image)
    fig.canvas.draw_idle()


def on_button_click(event):

    z_noise = torch.randn(1, 100, 1, 1).to(device)
    
    for i in range(100):
        slider_list[i].set_val(z_noise[0, i, 0, 0].item())

    fig.canvas.draw_idle()


for i in range(100):

    slider_list[i].on_changed(lambda x, idx=i: update(x, idx))


button.on_clicked(on_button_click)
plt.show()