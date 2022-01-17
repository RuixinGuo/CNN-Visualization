from model import Model
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

if __name__ == '__main__':
    '''Load the mnist_0.99 model we have trained using train_gpu.py.'''
    model = torch.load("models/mnist_0.99.pkl")
    model.eval()
    #print(model)
    #print(model.conv1)

    matplotlib.use('TkAgg')  # X11 back-end for displaying images from a remote machine.

    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=256)

    data_iter = iter(test_loader)
    test_x, test_label = data_iter.next()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    test_x = test_x.to(device)

    '''Register the hooks for the layers you want to output.'''
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.relu1.register_forward_hook(get_activation('relu1'))
    model.pool1.register_forward_hook(get_activation('pool1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    model.relu2.register_forward_hook(get_activation('relu2'))
    model.pool2.register_forward_hook(get_activation('pool2'))
    model.relu3.register_forward_hook(get_activation('relu3'))
    model.relu4.register_forward_hook(get_activation('relu4'))
    model.relu5.register_forward_hook(get_activation('relu5'))

    '''Forward propagation with an image (handwritten "5" for example)'''
    predict = model(test_x[8:9, 0:1, :, :].float())

    '''Visualize input image''' 
    print(predict)
    plt.imshow(test_x[8][0].detach().cpu(), cmap='gray')
    #plt.axis('off')
    plt.savefig('fig/input_img.png')
    plt.show()

    '''Visualize conv1 output'''
    t = activation['conv1'].size()
    print(t)
    for i in range(t[0]):
        for j in range(t[1]):
            k = i * t[1] + j + 1
            plt.subplot(t[0]*2, t[1]/2, k), plt.imshow(activation['conv1'][i][j].detach().cpu())
    plt.savefig('fig/conv1_output.png')
    plt.show()
    
    '''Visualize relu1 output'''
    t = activation['relu1'].size()
    for i in range(t[0]):
        for j in range(t[1]):
            k = i * t[1] + j + 1
            plt.subplot(t[0]*2, t[1]/2, k), plt.imshow(activation['relu1'][i][j].detach().cpu())
    plt.savefig('fig/relu1_output.png')
    plt.show()

    '''Visualize pool1 output'''
    t = activation['pool1'].size()
    for i in range(t[0]):
        for j in range(t[1]):
            k = i * t[1] + j + 1
            plt.subplot(t[0]*2, t[1]/2, k), plt.imshow(activation['pool1'][i][j].detach().cpu())
    plt.savefig('fig/pool1_output.png')
    plt.show()

    '''Visualize conv2 output'''
    t = activation['conv2'].size()
    for i in range(t[0]):
        for j in range(t[1]):
            k = i * t[1] + j + 1
            plt.subplot(t[0]*4, t[1]/4, k), plt.imshow(activation['conv2'][i][j].detach().cpu())
            plt.axis('off')
    plt.savefig('fig/conv2_output.png')
    plt.show()

    '''Visualize relu2 output'''
    t = activation['relu2'].size()
    for i in range(t[0]):
        for j in range(t[1]):
            k = i * t[1] + j + 1
            plt.subplot(t[0]*4, t[1]/4, k), plt.imshow(activation['relu2'][i][j].detach().cpu())
            plt.axis('off')
    plt.savefig('fig/relu2_output.png')
    plt.show()

    '''Visualize pool2 output'''
    t = activation['pool2'].size()
    for i in range(t[0]):
        for j in range(t[1]):
            k = i * t[1] + j + 1
            plt.subplot(t[0]*4, t[1]/4, k), plt.imshow(activation['pool2'][i][j].detach().cpu())
            plt.axis('off')
    plt.savefig('fig/pool2_output.png')
    plt.show()

    '''Visualize relu3 output'''
    plt.imshow(activation['relu3'].detach().cpu()).axes.get_yaxis().set_visible(False)
    plt.savefig('fig/relu3_output_.png')
    plt.show()
    '''Visualize relu4 output'''
    plt.imshow(activation['relu4'].detach().cpu()).axes.get_yaxis().set_visible(False)
    plt.savefig('fig/relu4_output.png')
    plt.show()
    '''Visualize relu5 output'''
    plt.imshow(activation['relu5'].detach().cpu()).axes.get_yaxis().set_visible(False)
    plt.savefig('fig/relu5_output.png')
    plt.show()
    
    '''Visualize conv1 filters'''
    t = model.conv1.weight.size()
    print(t)
    for i in range(t[0]):
        for j in range(t[1]):
            k = i * t[1] + j + 1
            plt.subplot(t[0]/3, t[1]*3, k), plt.imshow(model.conv1.weight[i][j].detach().cpu())
    plt.savefig('fig/conv1_weight.png')
    plt.show()

    '''Visualize conv2 filters'''
    t = model.conv2.weight.size()
    for i in range(t[0]):
        for j in range(t[1]):
            k = i * t[1] + j + 1
            plt.subplot(t[0], t[1], k), plt.imshow(model.conv2.weight[i][j].detach().cpu())
            plt.axis('off')
    plt.savefig('fig/conv2_weight.png')
    plt.show()

