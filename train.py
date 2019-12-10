import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

### Define training method
def train(n_epochs, loaders, model, optimizer, criterion, device, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        accuracy = 0
        start = time.time()
        ###################
        # train the model #
        ###################
        print('Epoch: {} - TRAIN'.format(epoch))
        model.train()
        with tqdm(total=len(loaders['train'])) as pbar:
            for batch_idx, (data, target) in enumerate(loaders['train']):
                train_start = time.time()
                # move to device
                data, target = data.to(device), target.to(device)
                # find the loss and update the model parameters accordingly
                optimizer.zero_grad()
                log_ps = model(data)
                loss = criterion(log_ps,target)
                loss.backward()
                optimizer.step()

                ## record the average training loss, using something like
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

                pbar.update(1)

        ######################
        # validate the model #
        ######################
        print('Epoch: {} - VALIDATE'.format(epoch))
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(loaders['valid'])) as pbar:
                for batch_idx, (data, target) in enumerate(loaders['valid']):
                    valid_start = time.time()
                    # move to device
                    data, target = data.to(device), target.to(device)
                    # forward pass and loss calculation
                    log_ps = model(data)
                    loss = criterion(log_ps,target)
                    # update the average validation loss
                    valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

                    ps = torch.exp(log_ps)
                    top_p,top_class = ps.topk(1,dim=1)
                    equals = top_class == target.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

                    pbar.update(1)

        # print training/validation statistics
        print('Training Loss: {:.6f} \tValidation Loss: {:.6f} \t Accuracy: {:.4f}%'.format(
            train_loss,
            valid_loss,
            (accuracy/len(loaders['valid']))*100
            ))
        print('Time Elapsed: {} seconds\n'.format(time.time()-start))

        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)

    # return trained model
    return model

if __name__ == "__main__":

    ### Load data
    data_dir = '../Files/dogImages'
    train_dir = os.path.normpath(data_dir+'/train')
    val_dir = os.path.normpath(data_dir+'/valid')
    test_dir = os.path.normpath(data_dir+'/test')

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize(225),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_val_transforms = transforms.Compose([transforms.Resize(225),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(val_dir, transform=test_val_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_val_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    loaders = {
        'train':trainloader,
        'valid':validloader,
        'test':testloader
    }

    ### Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Initialize pre-trained VGG16 model
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    ### Build and set classifier for model
    classifier = nn.Sequential(nn.Linear(25088, 3072),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(3072, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(1024, 306),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(306, 133),
                                   nn.LogSoftmax(dim=1))

    model.classifier = classifier
    model.to(device)

    ### Define loss function
    criterion = nn.CrossEntropyLoss()

    ### Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # train the model
    model = train(10, loaders, model, optimizer,
                          criterion, device, '../Files/model_trained.pt')
