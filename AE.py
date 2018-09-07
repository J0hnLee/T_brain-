import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='epoch number')
parser.add_argument('--batch_size', type=str, default=39, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dataset_path', type=str, default='new.npy', help='training dataset path')
parser.add_argument('--gpu_num', type=str, default='0', help='gpu devices number')
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_num

# if not os.path.exists('./mlp_img'):
#     os.mkdir('./mlp_img')

# def to_img(x):
#     x = 0.5*(x+1)
#     x = x.clamp(0, 1)
#     x = x.view(x.size(0), 1, 28, 28)
#     return x

data = np.load(opt.dataset_path)
data = data.astype('float32')
data = data.reshape((351273, 18*4))


num_epochs = opt.epochs
batch_size = opt.batch_size
learning_rete = opt.lr

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# dataset = MNIST('../data', transform=img_transform)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4*18, 36),
            nn.ReLU(True),
            nn.Linear(36, 18),
            nn.ReLU(True),
            nn.Linear(18, 10),
            # nn.ReLU(True),
            # nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 18),
            nn.ReLU(True),
            nn.Linear(18, 36),
            nn.ReLU(True),
            nn.Linear(36, 4*18),
            # nn.ReLU(True),
            # nn.Linear(128, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

if __name__ == "__main__":
    model = autoencoder()
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rete,
                                 weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in dataloader:
            # img, _ = data
            # img = img.view(img.size(0), -1)
            # img = Variable(img).cuda()

            d = data
            output = model(d)
            loss = criterion(output, d)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data.item()))
        # if epoch % 10 == 0:
        #     pic = to_img(output.cpu().data)
        #     save_image(pic, './mlp_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './sim_autoencoder.pth')
