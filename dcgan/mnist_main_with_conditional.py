'''
To do:
- Modify generator to accept z+C
- Modify discriminator similarly
'''

from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help='cifar10 | lsun | imagenet | folder | lfw | mnist')
parser.add_argument('--dataroot', default='../mnist_data/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='output/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for conditional')
parser.add_argument('--dim_convolve_feature_map', type=int, default=200, help='dimension of learned image feature map in discriminator')
parser.add_argument('--final_discriminator_dim', type=int, default=100, help='dimension of final hidden vector ')
opt = parser.parse_args()
#print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
#print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

#if torch.cuda.is_available() and not opt.cuda:
#    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
#nc = 3
nc = 1


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz+opt.n_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
#print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.convolve_image = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, opt.dim_convolve_feature_map, 4, 1, 0, bias=False),
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

        self.mlp_final = nn.Sequential(
            nn.Linear(opt.n_classes+opt.dim_convolve_feature_map,1, bias=False),
            nn.Sigmoid()
        )

    def forward(self,image_input,conditional_input,batch_size):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, image_input, range(self.ngpu))
        
        else:
            #get convolved feature map of images
            image_output = self.convolve_image(image_input)

            #concatenate MLP(one-vectors) with convolved feature map and pass through a final MLP
            final_output = self.mlp_final(torch.cat([image_output.view(batch_size,opt.dim_convolve_feature_map),conditional_input],1))

        return final_output

netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
#print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
conditional_input = torch.FloatTensor(opt.batchSize, opt.n_classes) #this matrix will hold the real class labels in one hot form
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
noise_conditionals = torch.FloatTensor(opt.batchSize, opt.n_classes) #matrix of one hots for randomly chosen class conditionals for fake examples
noise_with_conditionals = torch.FloatTensor(opt.batchSize, nz+opt.n_classes) #input to generator, noise concat with conditionals

#using fixed noise vectors + class conditionals, we will generate 50 fake images (5 for each class)
fixed_noise = torch.FloatTensor(50, nz).normal_(0, 1) #used for generating images
one_hot_labels = torch.LongTensor([i for i in range(10) for j in range(5)]).resize_(50,1) #0 five times, 1 five times, ... 9 five times.
one_hots = torch.zeros(50,opt.n_classes).scatter_(1, one_hot_labels, 1) #also used for generating images
fixed_noise_with_conditionals = torch.cat([fixed_noise,one_hots],1)
fixed_noise_with_conditionals.resize_(50, nz+opt.n_classes,1,1)

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    conditional_input = conditional_input.cuda()
    noise, fixed_noise_with_conditionals = noise.cuda(), fixed_noise_with_conditionals.cuda()
    noise_conditionals, noise_with_conditionals = noise_conditionals(), noise_with_conditionals()

input = Variable(input)
conditional_input = Variable(conditional_input) #declare conditional_input as variable
label = Variable(label)
noise = Variable(noise)
noise_conditionals = Variable(noise_conditionals)
noise_with_conditionals = Variable(noise_with_conditionals)
fixed_noise_with_conditionals = Variable(fixed_noise_with_conditionals)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        
        '''Train with real images and "real" labels'''
        netD.zero_grad()
        real_cpu, real_class_conditionals = data

        #size of current batch
        batch_size = real_cpu.size(0)

        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label)

        #resize real_class_conditional, vector of class labels - needed for scatter operation
        real_class_conditionals.resize_(batch_size,1) 
        #resize conditional_input to have (batch_size, n_classes), fill with 0s and then do one-hot operation
        conditional_input.data.resize_(batch_size, opt.n_classes).fill_(0).scatter_(1, real_class_conditionals, 1)

        output = netD(input,conditional_input,batch_size)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        '''Train with fake images and "fake" labels'''
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1) #nbatch number of noisy vectors of length nz

        #pick random labels for noise vectors for noise_conditionals
        random_noise_labels = torch.LongTensor(batch_size,1).random_(opt.n_classes)
        #fill in matrix of one-hots for the discriminator (fed in separately from fake images)
        noise_conditionals.data.resize_(batch_size, opt.n_classes).fill_(0).scatter_(1, random_noise_labels, 1)
        #concatenate with noise vectors for the generator (fed in simultaneously with noise vectors)
        noise_with_conditionals.data.resize_(batch_size, nz+opt.n_classes).copy_(torch.cat([noise.data,noise_conditionals.data],1))
        noise_with_conditionals.data.resize_(batch_size, nz+opt.n_classes,1,1)

        #Generate fake images
        fake = netG(noise_with_conditionals)
        
        label.data.fill_(fake_label)
        output = netD(fake.detach(),noise_conditionals,batch_size)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake,noise_conditionals,batch_size)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

    vutils.save_image(real_cpu, '%s/real_samples.png' % opt.outf, normalize=True)
    fake = netG(fixed_noise_with_conditionals)
    vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch), normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.m' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.m' % (opt.outf, epoch))
