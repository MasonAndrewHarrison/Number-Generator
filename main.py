import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

 
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)



device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 3e-4
z_dimensions = 64
image_dim = 28 * 28 * 1
batch_size = 64
num_epochs = 100

disc_model = Discriminator(image_dim).to(device)
gen_model = Generator(z_dimensions, image_dim).to(device)

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)


dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc_model.parameters(), lr=learning_rate)
opt_gen = optim.Adam(gen_model.parameters(), lr=learning_rate)

criterion = nn.BCELoss()

number = 0

for epoch in range(num_epochs):
    for i, (real,_) in enumerate(loader):

        z_noise = torch.randn(batch_size, z_dimensions).to(device)
        real_image = real.view(-1, 784).to(device)

        fake_image = gen_model(z_noise)
        fake_prediction = disc_model(fake_image.detach()).view(-1)
        real_prediction = disc_model(real_image).view(-1)

        
        #back prob for Discriminator
        disc_loss_fake = criterion(fake_prediction, torch.zeros_like(fake_prediction))
        disc_loss_real = criterion(real_prediction, torch.ones_like(real_prediction))


        disc_loss = (disc_loss_fake + disc_loss_real) / 2
        disc_model.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        """
        disc_model.zero_grad()
        disc_loss_fake.backward()
        opt_disc.step()

        disc_model.zero_grad()
        disc_loss_real.backward()
        opt_disc.step()
        """

        """
        disc_loss = -(torch.log(real_prediction).mean() + torch.log(1 - fake_prediction).mean())
        disc_model.zero_grad()
        disc_loss.backward()
        opt_disc.step()"""


        #back prob for Generator Model  
        gen_prediction = disc_model(fake_image).view(-1)
        gen_loss = -torch.log(gen_prediction).mean()
        gen_model.zero_grad()
        gen_loss.backward()
        opt_gen.step()
    
        if i == 0:
            print(f"{number} | Gen Loss: {gen_loss:.4f} | Disc Loss: {disc_loss_fake:.4f} & {disc_loss_real:.4f}")
            number += 1
            if number == 98:
        
                fake_image = fake_image[0].view(28, 28).detach().cpu()
                plt.imshow(fake_image, cmap='gray')
                plt.show()


