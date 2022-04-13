import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

import argparse
import csv
    
def test(model, test_loader, criterion, device):
    '''
    This function takes a model and a testing data loader and will get the test accuray/loss of the model
    '''    
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}")    
    print(f"Testing Loss: {total_loss}")    

def train(model, image_dataset_loaders, criterion, optimizer, device):
    '''
    This function takes a model and data loaders for training and will get train the model
    '''    
    epochs=2
    best_loss=1e6
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset_loaders[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 10  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset_loaders[phase].dataset),
                            100.0 * (running_samples / len(image_dataset_loaders[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                if running_samples>(0.1*len(image_dataset_loaders[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
    return model
    
def create_pretrained_model():
    '''
    Create pretrained resnet50 model
    When creating our model we need to freeze all the convolutional layers which we do by their requires_grad() attribute to False. 
    We also need to add a fully connected layer on top of it which we do use the Sequential API.
    '''
    model = models.resnet50(pretrained=True, progress=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model

class DogBreedDataset(Dataset):
    def __init__(self, annotations_file, base_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.base_dir = base_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.img_labels.iloc[idx, 1])
        image = read_image(img_path)
        label = int(self.img_labels.iloc[idx, 0])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def load_data():
    print('Downloading data')
    url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip'
    os.system(f"wget -c --read-timeout=5 --tries=0 {url}")    
    
    print('Download successful. Unzipping data')
    os.system(f"unzip -n -q dogImages.zip")
    print('Unzipping succesfull')

def create_metadata(database_path):
    with open(os.path.join(database_path, 'meta.csv'), 'w', encoding='UTF8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Id', 'Filename'])
        for root, dirs, files in os.walk(database_path):
            files.sort()
            for file in files:
                if file.lower().endswith('.jpg'):
                    classification_id = int(root.split("/")[2].split(".")[0])-1
                    rel_path = os.path.join(root, file)
                    row = [classification_id, rel_path]
                    writer.writerow(row)
    print('Creating metadata completed for file', os.path.join(database_path, 'meta.csv'))
      
def main(args):
    '''
    Initialize pretrained model
    '''
    model=create_pretrained_model()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    '''
    Create data loaders
    '''
    load_data()
    create_metadata('dogImages/test')
    create_metadata('dogImages/train')
    create_metadata('dogImages/valid')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_transform = transforms.Compose([
        transforms.Resize([256, ]),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std)])
    train_data = DogBreedDataset(annotations_file = 'dogImages/train/meta.csv', base_dir = '.', transform = image_transform)
    test_data = DogBreedDataset(annotations_file = 'dogImages/test/meta.csv', base_dir = '.', transform = image_transform)
    valid_data = DogBreedDataset(annotations_file = 'dogImages/valid/meta.csv', base_dir = '.', transform = image_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    
    image_dataset_loaders={'train':train_loader, 'valid':valid_loader}
    
    '''
    Create loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    Call the train function to start training model
    '''
    model=train(model, image_dataset_loaders, loss_criterion, optimizer, device)
    
    '''
    Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    Save the trained model
    '''
    torch.save(model, "dog_breed.pt")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    All the hyperparameters needed to use to train your model.
    '''
    # Training settings
    parser = argparse.ArgumentParser(description="Udacity AWS ML project 3 - HPO tuning")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    args = parser.parse_args()
    
    main(args)

# Example of using HPO tuning in script (from the course)
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import transforms
# from torchvision.datasets import MNIST

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output


# def train(model, train_loader, optimizer, epoch):
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print(
#                 "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                     epoch,
#                     batch_idx * len(data),
#                     len(train_loader.dataset),
#                     100.0 * batch_idx / len(train_loader),
#                     loss.item(),
#                 )
#             )


# def test(model, test_loader):
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print(
#         "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
#             test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
#         )
#     )


# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=64,
#         metavar="N",
#         help="input batch size for training (default: 64)",
#     )
#     parser.add_argument(
#         "--test-batch-size",
#         type=int,
#         default=1000,
#         metavar="N",
#         help="input batch size for testing (default: 1000)",
#     )
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=2,
#         metavar="N",
#         help="number of epochs to train (default: 14)",
#     )
#     parser.add_argument(
#         "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
#     )
#     args = parser.parse_args()

#     train_kwargs = {"batch_size": args.batch_size}
#     test_kwargs = {"batch_size": args.test_batch_size}

#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#     )
#     MNIST.mirrors = ["https://sagemaker-sample-files.s3.amazonaws.com/datasets/image/MNIST/"]
#     dataset1 = MNIST("../data", train=True, download=True, transform=transform)
#     dataset2 = MNIST("../data", train=False, transform=transform)
#     train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
#     test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

#     model = Net()

#     optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

#     for epoch in range(1, args.epochs + 1):
#         train(model, train_loader, optimizer, epoch)
#         test(model, test_loader)
    
#     torch.save(model.state_dict(), "mnist_cnn.pt")


# if __name__ == "__main__":
#     main()    

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torchvision
# from torchvision import datasets, models, transforms
# import time # for measuring time for testing, remove for students

# def test(model, test_loader, criterion, device):
#     print("Testing Model on Whole Testing Dataset")
#     model.eval()
#     running_loss=0
#     running_corrects=0
    
#     for inputs, labels in test_loader:
#         inputs=inputs.to(device)
#         labels=labels.to(device)
#         outputs=model(inputs)
#         loss=criterion(outputs, labels)
#         _, preds = torch.max(outputs, 1)
#         running_loss += loss.item() * inputs.size(0)
#         running_corrects += torch.sum(preds == labels.data).item()

#     total_loss = running_loss / len(test_loader.dataset)
#     total_acc = running_corrects/ len(test_loader.dataset)
#     print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    
# def train(model, train_loader, validation_loader, criterion, optimizer, device):
#     epochs=2
#     best_loss=1e6
#     image_dataset={'train':train_loader, 'valid':validation_loader}
#     loss_counter=0
    
#     for epoch in range(epochs):
#         for phase in ['train', 'valid']:
#             print(f"Epoch {epoch}, Phase {phase}")
#             if phase=='train':
#                 model.train()
#             else:
#                 model.eval()
#             running_loss = 0.0
#             running_corrects = 0
#             running_samples=0

#             for step, (inputs, labels) in enumerate(image_dataset[phase]):
#                 inputs=inputs.to(device)
#                 labels=labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)

#                 if phase=='train':
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#                 _, preds = torch.max(outputs, 1)
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data).item()
#                 running_samples+=len(inputs)
#                 if running_samples % 2000  == 0:
#                     accuracy = running_corrects/running_samples
#                     print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
#                             running_samples,
#                             len(image_dataset[phase].dataset),
#                             100.0 * (running_samples / len(image_dataset[phase].dataset)),
#                             loss.item(),
#                             running_corrects,
#                             running_samples,
#                             100.0*accuracy,
#                         )
#                     )
                
#                 #NOTE: Comment lines below to train and test on whole dataset
#                 if running_samples>(0.2*len(image_dataset[phase].dataset)):
#                     break

#             epoch_loss = running_loss / running_samples
#             epoch_acc = running_corrects / running_samples
            
#             if phase=='valid':
#                 if epoch_loss<best_loss:
#                     best_loss=epoch_loss
#                 else:
#                     loss_counter+=1

#         if loss_counter==1:
#             break
#     return model

# def create_model():
#     model = models.resnet18(pretrained=True)

#     for param in model.parameters():
#         param.requires_grad = False   

#     num_features=model.fc.in_features
#     model.fc = nn.Sequential(
#                    nn.Linear(num_features, 10))
#     return model

# batch_size=10
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Running on Device {device}")

# training_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# testing_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#         download=True, transform=training_transform)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#         shuffle=True)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#         download=True, transform=testing_transform)

# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#         shuffle=False)

# model=create_model()
# model=model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# train(model, trainloader, testloader, criterion, optimizer, device)

# test(model, testloader, criterion, device)