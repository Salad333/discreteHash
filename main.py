# created by hongyu yang
# 30-05-2019

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

import clustering

encode_length = 12
batch_size = 32
num_classes = 10
learning_rate = 0.001
epoch_lr_decrease = 30
k = 10000
num_epochs = 50


class CNN(nn.Module):
    def __init__(self, encoded_length):
        super(CNN, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.fc_encode = nn.Linear(4096, encoded_length)

    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x_pca = self.vgg.classifier(x)
        x = self.fc_encode(x_pca)
        return x_pca, x


class NetworkLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, b, u):
        return torch.mean(torch.abs(torch.pow((b - u), 2)))


cnn = CNN(encoded_length=encode_length)
cnn.cuda()
# cnn.load_state_dict(torch.load('temp.pkl'))


# Loss and Optimizer for the update CNN parameter U
criterion = NetworkLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, cnn.parameters()), lr=learning_rate, momentum=0.9,
                            weight_decay=5e-4)


def adjust_learning_rate(opt, epoch):
    lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for param_group in opt.param_groups:
        param_group['lr'] = lr


def main():
    # fix random seeds
    global train_dataloader, train_dataset
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    np.random.seed(31)

    # creating checkpoint repo
    exp_check = os.path.join('/home/hongyuyang/data/march/test/exp', 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # pre-processing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    dataset = datasets.ImageFolder('/home/hongyuyang/data/march/PIC/cifar', transform=transforms.Compose(tra))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             pin_memory=True)

    # clustering algorithm to use
    deepcluster = clustering.__dict__['Kmeans'](k)

    # Label Y: batchY; Bit B: batchB; Center C: batchC; CNN feature U: batchU
    # Train the Model
    for epoch in range(0, num_epochs):
        cnn.cuda().train()
        adjust_learning_rate(optimizer, epoch)
        # get the pca cnn feature from
        x_pca = compute_features(dataloader, cnn, len(dataset))
        clustering_loss = deepcluster.cluster(x_pca, verbose='store_true')
        # assign pseudo-labels
        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                  dataset.imgs)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True, )
        for i, (images, labels) in enumerate(train_dataloader):
            # should be N
            images = Variable(images.cuda())
            # should be M * N
            labels = Variable(labels.cuda())

            # initialize B, C ,U
            # should be K * N
            image_var = torch.autograd.Variable(images.cuda(), volatile=True)
            batchU =cnn(image_var)[1].cpu().detach().numpy()
            # Todo: replace with numpy here for now, might be better methods
            batchB = np.sign(batchU)
            # Todo: wrong make sure what is the meaning of mean
            batchC =batchU

            # Fix B,U, update C
            faieal = 0.1
            labels_var = torch.autograd.Variable(labels.cuda(), volatile=True)
            batchY = labels_var.cpu().numpy()
            # should be M * K
            # Todo: change it
            Q = batchY * np.transpose(batchB)
            # should be (M-1) * 1
            onesm = np.ones(batchY.shape[0] - 1)

            # Todo: change to 10 when the process finish
            # different as matlab : start at 0
            for time in range(2):
                z0 = batchC
                # M, for cifar10 is 10
                for m in range(batchY.size()[0]):
                    batchCk = np.delete(batchC, m, axis=0)
                    vkk = batchY[m]
                    batchYk = np.delete(batchY, m, axis=0)
                    # should be K * 1 inside this formula
                    batchC[m] = np.transpose(
                        np.sign(np.transpose(Q[m]) - np.transpose(batchCk) * batchYk * np.transpose(
                            vkk) - faieal * np.transpose(
                            batchCk) * onesm))
                if np.linalg.norm(batchC - z0, 'fro') < 1e-6 * np.linalg.norm(z0, 'fro'):
                    break

            # Fix C,U update B
            muule = 0.1
            batchB = np.sign(muule * np.transpose(batchC) * batchY + batchU)

            # Fix B, C. update U using SGD
            optimizer.zero_grad()
            # change back to pytorch
            batchU = torch.from_numpy(batchU).cuda()
            batchB = torch.from_numpy(batchB).cuda()
            net_loss = criterion(batchU, batchB)
            net_loss.backward()
            optimizer.step()

            if (i + 1) % (len(train_dataset) // batch_size / 2) == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f '
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                         net_loss))

    # Save the Trained Model
    torch.save(cnn.state_dict(), 'cifar.pkl')


def compute_features(dataloader, model, N):
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var)[0].cpu().numpy()
        if i == 0:
            features = np.zeros((N, aux.shape[1])).astype('float32')

        if i < len(dataloader) - 1:
            features[i * batch_size: (i + 1) * batch_size] = aux.astype('float32')
        else:
            # special treatment for final batch
            features[i * batch_size:] = aux.astype('float32')
    return features


if __name__ == '__main__':
    main()
