# -*- coding:utf-8 -*-
from __future__ import print_function

import copy

from PIL import Image
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.models as models

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.set_device(1)
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
imsize = 512 if USE_CUDA else 128

'''
unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

plt.figure()
imshow(style_img.data, title='Style Image')

plt.figure()
imshow(content_img.data, title='Content Image')
'''

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss



class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size() # batch_size, num_of_feature_maps, dimensions of map
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, style_img, content_img,
            style_weight=1000, content_weight=1,
            content_layers=content_layers_default,
            style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    content_losses = []
    style_losses = []

    model = nn.Sequential()
    gram = GramMatrix()

    if USE_CUDA:
        model = model.cuda()
        grma = gram.cuda()

    i = 1 
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = 'relu_' + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module('conten_loss_' + str(i), content_loss)
                contetn_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)

    return model, style_losses, content_losses

def get_input_param_optimizer(input_img):

    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, 
        num_steps=300, style_weight=1000, content_weight=1):
    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, style_img, content_img, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:

        def closure():

            input_param.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score = sl.backward()
            for cl in content_losses:
                content_score = cl.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print('run {}:'.format(run))
                print('Style Loss: {:4f} Content Loss: {:4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()

            return style_score + content_score
        optimizer.step(closure)

    input_param.data.clamp_(0, 1)

    return input_param.data

def image_loader(image_name):
    loader = transforms.Compose([
            transforms.Scale(imsize),
            transforms.ToTensor()]
    )
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image

def imshow(tensor, title=None):
    image = tensor.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def save_tensor_as_image(tensor, filename='result.png'):
    image = tensor.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = transforms.ToPILImage()(image)
    image.save(filename)

if __name__ == '__main__':
    print('Loading vgg19...')
    cnn = models.vgg19(pretrained=True).features
    if USE_CUDA:
        cnn = cnn.cuda()

    style_img = image_loader('images/picasso.jpg').type(dtype)
    content_img = image_loader('images/dancing.jpg').type(dtype)

    assert style_img.size() == content_img.size(), 'we need to import style and content images of the same size'

    input_img = content_img.clone()

    output = run_style_transfer(cnn, content_img, style_img, input_img)
    filename = 'transfer.png'
    print('Save transferred image into file %s' % filename)
    save_tensor_as_image(output, filename)