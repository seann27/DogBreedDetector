import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import sys
import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from torchvision import datasets

# # grab file from ajax call
# import cgi, cgitb
# cgitb.enable()
# form = cgi.FieldStorage()
# if form.has_key("file"):
#     file = form["file"].value

test_val_transforms = transforms.Compose([transforms.Resize(225),
					  transforms.CenterCrop(224),
					  transforms.ToTensor(),
					  transforms.Normalize([0.485, 0.456, 0.406],
										  [0.229, 0.224, 0.225])])

data_dir = os.path.normpath("../Files/dogImages")
train_dir = os.path.normpath(data_dir+'/train')
train_data = datasets.ImageFolder(train_dir)

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in train_data.classes]

# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# grab file from commandline arguments
import argparse
parser = argparse.ArgumentParser(description="File upload")
parser.add_argument('image', help='Filepath of image')
parser.add_argument('--name', help='Name of image')
args = parser.parse_args()
file = os.path.normpath(args.image)
if args.name:
	name = args.name
else:
	name = file

# terminate program if image file cannot be found
if os.path.exists(file) is False:
	print("Could not find file "+file)
	exit()

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

VGG16 = models.vgg16(pretrained=True)
VGG16.to(device)
def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    img_path = os.path.normpath(img_path)
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    tr = transforms.Compose([transforms.Resize(225),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])
    np_tensor = process_image(img_path,tr)
    np_tensor.unsqueeze_(0)
    np_tensor = np_tensor.float()
    np_tensor = np_tensor.to(device)
    log_ps = VGG16.forward(np_tensor)
    ps = torch.exp(log_ps)
    top_p,top_class = ps.topk(1,dim=1)

    ## Return the *index* of the predicted class for that image

    return int(top_class) # predicted class index

def dog_detector(img_path):
    ## TODO: Complete the function.
    index = VGG16_predict(img_path)

    return index >= 151 and index <= 268

def generate_image_map(dog_files):
	image_map = {}
	for file in dog_files:
		name = os.path.basename(os.path.dirname(file))
		name = name.split('.')[1]
		if name in image_map.keys():
			image_map[name].append(file)
		else:
			image_map[name] = [file]
	return image_map

# returns image tensor
def process_image(image_path,transformation):
	image = Image.open(image_path)
	img = transformation(image)
	return img

def get_breed_image(prediction,image_map):
	pred = prediction.replace(" ","_")
	images = image_map[pred]
	img = random.choice(images)
	return img

def predict_breed_transfer(img_path,model,transforms):
	# load the image and return the predicted breed
	np_tensor = process_image(img_path,transforms)
	np_tensor.unsqueeze_(0)
	np_tensor = np_tensor.float()
	np_tensor = np_tensor.to(device)
	output = model(np_tensor)
	ps = torch.exp(output)
	top_p,top_class = ps.topk(4,dim=1)
	top_p = top_p.cpu().detach().numpy().reshape(-1)
	top_class = top_class.cpu().numpy().reshape(-1)
	return top_p,top_class

def normalize_predictions(top_p,top_class):
	norm_preds = {}
	for p,c in zip(top_p,top_class):
		total = top_p.sum()
		norm_preds[c] = p/total
	return norm_preds

def filter_matches(matches):
	filtered = {'Other':0}
	for key,val in matches.items():
		if val*100 >= 5:
			filtered[class_names[key]] = val
		else:
			filtered['Other'] += val
	return filtered

def plot_figures(file, best_match, comparison, matches, text):
    my_imgs = [Image.open(file),Image.open(comparison)]
    my_labels = ['Your image','{}\n{:.2f}% match'.format(best_match,matches[best_match]*100)]

    for key,val in matches.items():
        if key != 'Other' and key != best_match:
            my_imgs.append(Image.open(get_breed_image(key,img_map)))
            my_labels.append('{}\n{:.2f}% match'.format(key,val*100))

    f = plt.figure(figsize=(8,8))
    f.suptitle(text, fontsize=16)
    ax1 = f.add_subplot(1,2,1)
    ax1.imshow(my_imgs[0])
    ax1.set_title(my_labels[0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = f.add_subplot(1,2,2)
    ax2.imshow(my_imgs[1])
    ax2.set_title(my_labels[1])
    ax2.set_xticks([])
    ax2.set_yticks([])
    f.subplots_adjust(top=1)

    if len(my_imgs) > 2:
        f2 = plt.figure(figsize=(6,6))
        f2.suptitle('Other resemblances', fontsize=12)
        for idx,i in enumerate(my_imgs[2:]):
            ax = f2.add_subplot(1,(len(my_imgs)-2),idx+1)
            ax.imshow(i)
            ax.set_title(my_labels[idx+2])
            ax.set_xticks([])
            ax.set_yticks([])
        f2.subplots_adjust(top=1)

    plt.show()

def predict(file,model,img_map,transforms):
	matches = {}
	for i in range(0,99):
		top_p,top_class = predict_breed_transfer(file,model,transforms)
		preds = normalize_predictions(top_p,top_class)
		for key,val in preds.items():
			if key in matches.keys():
				matches[key] += val
			else:
				matches[key] = val
	best = 0
	best_match = ''
	for key,val in matches.items():
		matches[key] = val/100
		if val > best:
			best = val
			best_match = class_names[key]
	comparison = get_breed_image(best_match,img_map)
	return best_match,comparison,filter_matches(matches)

# build and initialize model
model = models.vgg16(pretrained=True)

for param in model.parameters():
	param.requires_grad = False

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

# load checkpoint file from training
try:
	checkpoint = os.path.normpath('../Files/model_trained.pt')
except:
	print("Model checkpoint not found. Please train model or move checkpoint file to ../Files/")
	exit()

# send checkpoint file data into model
model.load_state_dict(torch.load(checkpoint,map_location=device))

# load filenames for human and dog images
human_files = np.array(glob(os.path.normpath("../Files/lfw/*/*")))
dog_files = np.array(glob(os.path.normpath("../Files/dogImages/*/*/*")))

img_map = generate_image_map(dog_files)

text = "Error, couldn't detect a dog or person"
if dog_detector(file):
    text = "Woof Woof Hello Doggy!"
elif face_detector(file) > 0:
    text = "Hello human! If you were a dog..."
else:
    print(text)
    exit()

best_match,comparison,matches = predict(file,model,img_map,test_val_transforms)
plot_figures(file,best_match,comparison,matches,text)
