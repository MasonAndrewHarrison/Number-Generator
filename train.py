import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import Discriminator, Generator
from torch.utils.tensorboard import SummaryWriter
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

z_dimensions = 100
image_dim = 64
batch_size = 64
num_epochs = 20


disc_model = Discriminator(1, 64).to(device)
gen_model = Generator(z_dimensions, 1, 64).to(device)

if os.path.exists("Generator_Weights.pth"):
    gen_model.load_state_dict(torch.load("Generator_Weights.pth", map_location=device))
    gen_model.to(device)

if os.path.exists("Discriminator_Weights.pth"):
    disc_model.load_state_dict(torch.load("Discriminator_Weights.pth", map_location=device))
    disc_model.to(device)


transforms = transforms.Compose(
    [
        transforms.Resize(image_dim),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
fixed_noise = torch.randn(32, z_dimensions, 1, 1).to(device)

opt_disc = optim.Adam(disc_model.parameters(), lr=1e-5, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen_model.parameters(), lr=1e-4, betas=(0.5, 0.999))

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

criterion = nn.BCELoss()

step = 0
for epoch in range(num_epochs):
    for i, (real,_) in enumerate(loader): 

        z_noise = torch.randn(batch_size, z_dimensions, 1, 1).to(device)
        real_image = real.to(device)

        fake_image = gen_model(z_noise)
        
        #back prob for Discriminator
        fake_prediction = disc_model(fake_image.detach()).view(-1)
        real_prediction = disc_model(real_image).view(-1)

        disc_loss_fake = criterion(fake_prediction, torch.zeros_like(fake_prediction))
        disc_loss_real = criterion(real_prediction, torch.ones_like(real_prediction))

        disc_loss = (disc_loss_fake + disc_loss_real) / 2
        disc_model.zero_grad()
        disc_loss.backward()
        opt_disc.step()


        #back prob for Generator Model  
        gen_prediction = disc_model(fake_image).view(-1)
        gen_loss = -torch.log(gen_prediction).mean()
        gen_model.zero_grad()
        gen_loss.backward()
        opt_gen.step()

    
        if i == 0:
            print(f"{epoch} | Gen Loss: {gen_loss:.4f} | Disc Loss: {disc_loss_fake:.4f} & {disc_loss_real:.4f}")
            
            torch.save(gen_model.state_dict(), "Generator_Weights.pth")
            torch.save(disc_model.state_dict(), "Discriminator_Weights.pth")

        if i % 500:
        
            with torch.no_grad():
                fake = gen_model(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real_image[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake_image[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1


