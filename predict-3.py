import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import os
#Command Line Arguments

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input', type=str, help='path to input image')
ap.add_argument('checkpoint', type=str, help='path to checkpoint')
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input
number_of_outputs = pa.top_k
power = pa.gpu
#input_img = pa.input_img
path = pa.checkpoint

def load_model(path):
    trained_model = torch.load(path)
    class_idx = trained_model['class_to_idx']
    load_model = models.vgg11(pretrained=True)
        
    for param in load_model.parameters():
        param.requires_grad = False
    
    load_model.classifier = trained_model['classifier']
    load_model.load_state_dict(trained_model['state_dict'])
    load_model.class_to_idx = trained_model['class_to_idx']
    return load_model

def load_data():
    '''
    Arguments : the datas' path
    Returns : The loaders for the train, validation and test datasets
    This function receives the location of the image files, applies the necessery transformations (rotations,flips,normalizations and crops) and converts the images to tensor in order to be able to be fed into the neural network
    '''

    data_dir = './flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

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


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # The data loaders are going to use to load the data to the NN(no shit Sherlock)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)



    return  train_data,trainloader , vloader, testloader
def process_image(image):
    img = Image.open(image) 
    make_img_good = transforms.Compose([ 
	        transforms.Resize(256),
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	    ])
    tensor_image = make_img_good(img)
    return tensor_image
	

def predict(path_image, model, topk,power='gpu'):
	  
	if torch.cuda.is_available() and power=='gpu':
	        model.to('cuda:0')
	
	img_torch = process_image(path_image)
	img_torch = img_torch.unsqueeze_(0)
	img_torch = img_torch.float()
	if power == 'gpu':
		with torch.no_grad():
	        	output = model.forward(img_torch.cuda())
	else:
	        with torch.no_grad():
	            output=model.forward(img_torch)
	probability = F.softmax(output.data,dim=1)
	return probability.topk(topk)
	
model=load_model(path)
probabilities = predict(path_image, model, number_of_outputs, power)
with open('cat_to_name.json', 'r') as json_file:
	    cat_to_name = json.load(json_file)
labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])
	

i=0
while i < number_of_outputs:
	print("{} with a probability of {}".format(labels[i], probability[i]))
	i += 1

