from model import Model
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    model = Model().to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    cost = CrossEntropyLoss()
    epoch = 100

    for _epoch in range(epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x, train_label = train_x.to(device), train_label.to(device)
            label_np = np.zeros((train_label.shape[0], 10))
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = cost(predict_y, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            sgd.step()

        correct = 0
        _sum = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x, test_label = test_x.to(device), test_label.to(device)
            predict_y = model(test_x.float()).detach().cpu()
            predict_ys = np.argmax(predict_y, axis=-1)
            test_label1 = test_label.cpu()
            label_np = test_label1.numpy()
            _ = predict_ys == test_label1
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        print('accuracy: {:.2f}'.format(correct / _sum))
        torch.save(model, 'models/mnist_{:.2f}.pkl'.format(correct / _sum))
