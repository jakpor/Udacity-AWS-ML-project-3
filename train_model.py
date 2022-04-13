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
import time

from smdebug import modes
from smdebug.pytorch import get_hook
import smdebug.pytorch as smd

def test(model, test_loader, criterion, device, hook):
    '''
    This function takes a model and a testing data loader and will get the test accuray/loss of the model
    '''    
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    hook.set_mode(modes.EVAL)
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

def train(model, image_dataset_loaders, criterion, optimizer, device, hook):
    '''
    This function takes a model and data loaders for training and will get train the model
    '''    
    epochs=3
    best_loss=1e6
    loss_counter=0
    
    epoch_times = []        
    for epoch in range(epochs):
        start = time.time()
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
                hook.set_mode(modes.TRAIN)                
            else:
                model.eval()
                hook.set_mode(modes.EVAL)                
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset_loaders[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                hook.register_loss(criterion)
                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                accuracy = running_corrects/running_samples
                print("Phase: {}, Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        phase,
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
        
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        print("Epoch %d: time %.1f sec" % (epoch, epoch_time))

        if loss_counter==1:
            break
            
    # calculate training time after all epoch
    p50 = np.percentile(epoch_times, 50)
    return p50
    
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
    os.system(f"unzip -o -q dogImages.zip")
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    model=create_pretrained_model()
    model.to(device)
    
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    '''
    Create debug hook
    '''
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    
    '''
    Call the train function to start training model
    '''
    median_time = train(model, image_dataset_loaders, loss_criterion, optimizer, device, hook)
    print("Median training time per Epoch=%.1f sec" % median_time)
    
    '''
    Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device, hook)
    
    '''
    Save the trained model
    '''
    print("Median training time per Epoch=%.1f sec" % median_time)
    torch.save(model.state_dict(), "model_state.pt")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    All the hyperparameters needed to use to train your model.
    '''
    # Training settings
    parser = argparse.ArgumentParser(description="Udacity AWS ML project 3 - Model training with debug")
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