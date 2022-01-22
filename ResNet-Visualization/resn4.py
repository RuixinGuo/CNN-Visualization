#from model import Model
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib
import json


#imagenet_test = datasets.ImageNet(
#        root="data",
#        train=False,
#        download=True,
#        transform=ToTensor(),
#        )


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.clone().detach()
        #activation[name] = output.detach()
    return hook

f = open('imagenet_class_index.json', "r")
imagenet_classes = json.loads(f.read())
#print(imagenet_classes[str(99)])
f.close()


transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])

dataset = datasets.ImageFolder('./dataset', transform=transform)
test_loader = DataLoader(dataset, batch_size=256)

data_iter = iter(test_loader)
test_x, test_label = data_iter.next()
#print(test_x[25:26, :, :, :])
#print(test_label[0])
print(test_x.size())
img = test_x[61]
img = img.swapaxes(0,1)
img = img.swapaxes(1,2)
plt.imshow(img)
plt.savefig('fig/input_img.png')
plt.show()

resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
print(resnet18)

resnet18.conv1.register_forward_hook(get_activation('conv1'))
resnet18.bn1.register_forward_hook(get_activation('bn1'))
resnet18.relu.register_forward_hook(get_activation('relu'))
resnet18.maxpool.register_forward_hook(get_activation('maxpool'))
resnet18.layer1[0].conv1.register_forward_hook(get_activation('layer1_0.conv1'))
resnet18.layer1[0].bn1.register_forward_hook(get_activation('layer1_0.bn1'))
#resnet18.layer1[0].relu.register_forward_hook(get_activation('layer1_0.relu'))
resnet18.layer1[0].conv2.register_forward_hook(get_activation('layer1_0.conv2'))
resnet18.layer1[0].bn2.register_forward_hook(get_activation('layer1_0.bn2'))
resnet18.layer1[0].register_forward_hook(get_activation('layer1_0'))
resnet18.layer1[1].conv1.register_forward_hook(get_activation('layer1_1.conv1'))
resnet18.layer1[1].bn1.register_forward_hook(get_activation('layer1_1.bn1'))
#resnet18.layer1[1].relu.register_forward_hook(get_activation('layer1_1.relu'))
resnet18.layer1[1].conv2.register_forward_hook(get_activation('layer1_1.conv2'))
resnet18.layer1[1].bn2.register_forward_hook(get_activation('layer1_1.bn2'))
resnet18.layer1[1].register_forward_hook(get_activation('layer1_1'))
predict = resnet18(test_x[61:62, :, :, :])

#print(activation['layer1_1.bn2'][0][0] + activation['layer1_0'][0][0])
#print(activation['layer1_1'][0][0])
print(activation['bn1'][0][2])

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['conv1'][0][i*2])
    plt.savefig('fig1/conv1.png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['bn1'][0][i*2])
    plt.savefig('fig1/bn1.png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['relu'][0][i*2])
    plt.savefig('fig1/relu.png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['maxpool'][0][i*2])
plt.savefig('fig1/maxpool.png')
plt.show()

'''
for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['layer1_0.bn2'][0][i*2])
plt.savefig('fig1/layer1_0.bn2.png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['layer1_0.bn2'][0][i*2] + activation['maxpool'][0][i*2])
plt.savefig('fig1/layer1_0.bn2+maxpool.png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['layer1_0'][0][i*2])
plt.savefig('fig1/layer1_0.png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['layer1_0'][0][i*2] - (activation['layer1_0.bn2'][0][i*2] + activation['maxpool'][0][i*2]))
plt.savefig('fig1/layer1_0-(layer1_0.bn2+maxpool).png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(torch.nn.ReLU()(activation['layer1_0.bn2'][0][i*2]) - activation['layer1_0.bn2'][0][i*2])
plt.savefig('fig1/relu(layer1_0.bn2)-layer1_0.bn2.png')
plt.show()


for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['layer1_0'][0][i*2])
plt.savefig('fig1/layer1_0.png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['layer1_1.bn2'][0][i*2])
plt.savefig('fig1/layer1_1.bn2.png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['layer1_1.bn2'][0][i*2] + activation['layer1_0'][0][i*2])
plt.savefig('fig1/layer1_1.bn2+layer1_0.png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['layer1_1'][0][i*2])
plt.savefig('fig1/layer1_1.png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(activation['layer1_1'][0][i*2] - (activation['layer1_1.bn2'][0][i*2] + activation['layer1_0'][0][i*2]))
plt.savefig('fig1/layer1_1-(layer1_1.bn2+layer1_0).png')
plt.show()

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(torch.nn.ReLU()(activation['layer1_1.bn2'][0][i*2]) - activation['layer1_1.bn2'][0][i*2])
plt.savefig('fig1/relu(layer1_1.bn2)-layer1_1.bn2.png')
plt.show()
'''

'''
plt.subplot(221), plt.imshow(activation['layer1_0'][0][1])
plt.subplot(222), plt.imshow(activation['layer1_1.bn2'][0][1])
plt.subplot(223), plt.imshow(activation['layer1_1.bn2'][0][1] + activation['layer1_0'][0][1])
plt.subplot(224), plt.imshow(activation['layer1_1'][0][1])
plt.savefig('fig/compare.png')
plt.show()
'''

'''
plt.subplot(121), plt.imshow(activation['layer1_1'][0][6] - (activation['layer1_1.bn2'][0][6] + activation['layer1_0'][0][6]))
plt.subplot(122), plt.imshow(torch.nn.ReLU()(activation['layer1_1.bn2'][0][6]) - activation['layer1_1.bn2'][0][6])
plt.show()
'''

'''
layer_name = ['layer1_0.conv1', 'layer1_0.bn1', 'layer1_0.conv2', 'layer1_0.bn2', 'layer1_0', 'layer1_1.conv1', 'layer1_1.bn1', 'layer1_1.conv2', 'layer1_1.bn2', 'layer1_1']

for i in range(10):
    for j in range(1, 17):
        plt.subplot(4, 4, j), plt.imshow(activation[layer_name[i]][0][j])
    plt.savefig('fig/'+layer_name[i]+'.png')
    plt.show()
'''

'''
plt.subplot(221), plt.imshow(activation['layer1_0'][0][0] - activation['layer1_0.bn2'][0][0])
plt.subplot(222), plt.imshow(activation['layer1_0.bn2'][0][0] - activation['layer1_0.conv2'][0][0])
plt.subplot(223), plt.imshow(activation['layer1_0'][0][0] - activation['layer1_0.relu'][0][0])
plt.subplot(224), plt.imshow(activation['layer1_1.bn2'][0][0] - activation['layer1_0.bn2'][0][0])
plt.show()

print(activation['layer1_1.bn1'][0][0])
'''

'''
print(predict)
top5 = torch.topk(predict, 5)
values, indices = top5
print(values)
print(indices)
for i in range(0, 5):
    #print(str(indices[0][i].numpy()))
    print(imagenet_classes[str(indices[0][i].numpy())])
'''
