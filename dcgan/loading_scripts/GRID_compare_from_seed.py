'''
Given the same noise vector z, condition on 10 different classes and show the output. 
Also show vanilla generation on the same noise vector.
'''

from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import numpy as np
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
parser.add_argument('--dataset', default='cifar10', help='cifar10 | lsun | imagenet | folder | lfw | mnist')
parser.add_argument('--dataroot', default='../../cifar10_data/', help='path to dataset')
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
parser.add_argument('--netG_cond', default='../cifar/conditional_400_5/netG_epoch_12.m', help="path to netG (to continue training)")
parser.add_argument('--netG', default='../cifar/baseline_normalized_40/netG_epoch_19.m', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for conditional')
parser.add_argument('--dim_convolve_feature_map', type=int, default=200, help='dimension of learned image feature map in discriminator')
parser.add_argument('--final_discriminator_dim', type=int, default=100, help='dimension of final hidden vector ')
opt = parser.parse_args()

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
if opt.dataset == 'mnist':
	nc = 1
else:
	nc = 3

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

'''
Bc we are loading both conditional and unconditional models in this script, 
and bc _netG must inherit from super(_netG) with same name, this script passes input_dim as a param to init fn.
-Set input_dim as nz for vanilla
-Set input_dim as nz+opt.n_classes for conditional
'''
class _netG(nn.Module):
    def __init__(self, ngpu, input_dim):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( input_dim, ngf * 8, 4, 1, 0, bias=False),
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

netG_cond = _netG(ngpu, input_dim = nz+opt.n_classes)
netG_cond.apply(weights_init)
netG_cond.load_state_dict(torch.load(opt.netG_cond))
netG_cond.eval()

netG = _netG(ngpu, input_dim = nz)
netG.apply(weights_init)
netG.load_state_dict(torch.load(opt.netG))
netG.eval()

num_examples = 3
z_np = np.random.randn(num_examples,nz) #different z vectors

z = torch.Tensor(z_np) #for vanilla model
z.resize_(num_examples,nz,1,1)

#make 10 (n_classes) copies of the first array, 10 copies of the second, etc.
z_repeating = torch.Tensor(np.array([i for i in z_np for j in range(opt.n_classes)]))

#create one hot matrix
one_hot_labels = torch.LongTensor([i for j in range(num_examples) for i in range(opt.n_classes)]).resize_(num_examples*opt.n_classes,1) #list of labels
one_hots = torch.zeros(num_examples*opt.n_classes,opt.n_classes).scatter_(1,one_hot_labels,1)

#concat z repeats with conditionals
z_repeating_with_conditionals = torch.cat([z_repeating,one_hots],1) #matrix of correspondng one hots
z_repeating_with_conditionals.resize_(num_examples*opt.n_classes,opt.nz+opt.n_classes,1,1)

z = Variable(z)
z_repeating_with_conditionals = Variable(z_repeating_with_conditionals)

fake = netG(z)
fake_conditional = netG_cond(z_repeating_with_conditionals)

vutils.save_image(fake.data, opt.outf+'baseline_normalized_fixed_noise_vanilla_fake.png', nrow=10, normalize=True)
vutils.save_image(fake_conditional.data, opt.outf+'conditional_400_fixed_noise_conditional_fake.png', nrow=10, normalize=True)


