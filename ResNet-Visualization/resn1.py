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
        activation[name] = output.detach()
    return hook

f = open('imagenet_class_index.json', "r")
imagenet_classes = json.loads(f.read())
f.close()


transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])

dataset = datasets.ImageFolder('./dataset', transform=transform)
test_loader = DataLoader(dataset, batch_size=256)

data_iter = iter(test_loader)
test_x, test_label = data_iter.next()

print(test_x.size())
img = test_x[61]
img = img.swapaxes(0,1)
img = img.swapaxes(1,2)
plt.imshow(img)
plt.savefig('fig/input_img.png')
plt.show()

resnet18 = models.resnet152(pretrained=True)
resnet18.eval()
print(resnet18)

resnet18.layer1.register_forward_hook(get_activation('layer1'))
resnet18.layer2.register_forward_hook(get_activation('layer2'))
resnet18.layer3.register_forward_hook(get_activation('layer3'))
resnet18.layer4.register_forward_hook(get_activation('layer4'))
predict = resnet18(test_x[61:62, :, :, :])

plt.subplot(221), plt.imshow(activation['layer1'][0][0])
plt.subplot(222), plt.imshow(activation['layer2'][0][0])
plt.subplot(223), plt.imshow(activation['layer3'][0][0])
plt.subplot(224), plt.imshow(activation['layer4'][0][0])
plt.savefig('fig/layer_img.png')
plt.show()

