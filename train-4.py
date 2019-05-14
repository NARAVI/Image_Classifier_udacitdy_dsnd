import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse


ap = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

ap.add_argument('data_dir', type=str, action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=12)
ap.add_argument('--arch', dest="arch", action="store", default="vgg11", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs



def load_data(where ):
    '''
    Arguments : the datas' path
    Returns : The loaders for the train, validation and test datasets
    This function receives the location of the image files, applies the necessery transformations (rotations,flips,normalizations and crops) and converts the images to tensor in order to be able to be fed into the neural network
    '''

    data_dir = where
    train_dir = data_dir+'/train'
    valid_dir = data_dir+'/valid'
    test_dir = data_dir+'/test'

    #Apply the required transfomations to the test dataset in order to maximize the efficiency of the learning
    #process


    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Crop and Resize the data and validation images in order to be able to be fed into the network

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.RandomResizedCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    print(train_dir)
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # The data loaders are going to use to load the data to the NN(no shit Sherlock)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)



    return  train_data,trainloader , vloader, testloader ,len(train_data) , len(validation_data)

# This method loads and tunes in a model
def load_model(arch='vgg11', num_labels=102, hidden_units=4096,dropout=0.5,lr = 0.003,power='gpu'):
    # Load a pre-trained model
    if arch=='vgg11':
        # Load a pre-trained model
        model = models.vgg11(pretrained=True)
    elif arch=='vgg16':
        model = models.vgg16(pretrained=True)
    elif arch=='vgg19':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError('Unexpected network architecture', arch)
        
    # Freeze its parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Features, removing the last layer
    features = list(model.classifier.children())[:-1]
  
    # Number of filters in the bottleneck layer
    num_filters = model.classifier[len(features)].in_features

    # Extend the existing architecture with new layers
    features.extend([
        nn.Dropout(dropout),
        nn.Linear(num_filters, hidden_units),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(True),
        nn.Linear(hidden_units, 102),
        ##nn.Softmax(dim=1) 
    ])
    
    model.classifier = nn.Sequential(*features) 
    #model.classifier = classifier
    criterion = nn.CrossEntropyLoss()

# Only train the classifier parameters, feature parameters are frozen
#optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    optimizer=optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.003, momentum=0.9)
    if torch.cuda.is_available() and power == 'gpu':
        model.cuda()

    return model, criterion, optimizer

def train_network(trainsize,validsize, model, criterion, optimizer, epochs = 12, print_every=20, loader='trainloader',vloader='v_loader', power='gpu'):
    '''
    Arguments: The model, the criterion, the optimizer, the number of epochs, teh dataset, and whether to use a gpu or not
    Returns: Nothing
    This function trains the model over a certain number of epochs and displays the training,validation and accuracy every "print_every" step using cuda if specified. The training method is specified by the criterion and the optimizer which are NLLLoss and Adam respectively
    '''
    steps = 0
    running_loss = 0

    print("--------------Training is starting------------- ")
    for e in range(epochs):
        running_loss = 0
        acc_train=0.0
        for ii, (inputs, labels) in enumerate(loader):
            steps += 1
            if torch.cuda.is_available() and power=='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.item()
            acc_train += torch.sum(preds == labels.data)
            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0


                for ii, (inputs2,labels2) in enumerate(vloader):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model.to('cuda:0')

                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        _, preds = torch.max(outputs.data, 1)
                        accuracy += torch.sum(preds == labels2.data) 
                vlost = vlost / validsize
                accuracy = float(accuracy)/validsize



                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/trainsize),
                      "Training Accuracy: {:.4f}".format(float(acc_train)/trainsize),
                      "Validation Lost {:.4f}".format(vlost),
                       "Validation Accuracy: {:.4f}".format(accuracy))


                running_loss = 0


    print("-------------- Finished training -----------------------")
    print("Dear User I the ulitmate NN machine trained your model. It required")
    print("----------Epochs: {}------------------------------------".format(epochs))
    print("----------Steps: {}-----------------------------".format(steps))
    print("That's a lot of steps")
    

def save_checkpoint(model,optimizer,traindata,path='checkpoint.pth',structure ='vgg11', hidden_layer1=4096,dropout=0.5,lr=0.003,epochs=12):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing
    This function saves the model at a specified by the user path
    '''
    model.class_to_idx = traindata.class_to_idx
    model.cpu
    torch.save({'structure' :structure,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx,
                'classifier': model.classifier,
                'optimizer': optimizer.state_dict()},
                path)

def load_checkpoint(path='checkpoint.pth'):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases
    '''
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']

    model,_,_ = load_model(structure , dropout,hidden_layer1,lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
 
traindata,trainloader, v_loader, testloader,traindata_size,validdata_size = load_data(where)


model, optimizer, criterion = load_model(structure,dropout,hidden_layer1,lr,power)


train_network(traindata_size,validdata_size,model, optimizer, criterion, epochs, 20, trainloader,v_loader, power)    


save_checkpoint(model,optimizer,traindata,path,structure,hidden_layer1,dropout,lr)

print("All Set and Done. The Model is trained") 





