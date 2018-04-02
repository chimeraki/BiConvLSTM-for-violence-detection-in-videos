from network import ConvLSTM,Violence_predictor
from Data_loader import VideoDataset,RandomCrop

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import random,torch
import numpy as np

cuda_use=1
seed=250

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if(cuda_use==1):
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    print("torch.backends.cudnn.enabled is: ", torch.backends.cudnn.enabled)

#train_transform=transforms.Compose([transforms.ToTensor()])
train_dataset=VideoDataset(root_dir='/vulcan/scratch/koutilya/violence_datasets/Hockey_dataset/train/',transform=None)
train_dataloader=DataLoader(train_dataset,batch_size=8,shuffle=True)

#test_transform=transforms.Compose([transforms.Resize((224)),transforms.ToTensor()])
test_dataset=VideoDataset(root_dir='/vulcan/scratch/koutilya/violence_datasets/Hockey_dataset/test/',transform=None)
test_dataloader=DataLoader(test_dataset,batch_size=4,shuffle=True)

network=Violence_predictor(cuda_use)
if(cuda_use==1):
	network=network.cuda()

print(network)
criterion=nn.CrossEntropyLoss()
optim=optim.Adam(network.parameters(),lr=0.0001)
epochs=100
display_step=1

for epoch in range(epochs):
	for i,data in enumerate(train_dataloader,0):
		if(i==1):
			break
		network.zero_grad()
		video,label=data
		if(cuda_use):
			video,label=Variable(video.cuda()),Variable(label.cuda())
		else:
			video,label=Variable(video),Variable(label)
		optim.zero_grad()
		output=network(video,10)
		#print(output.shape)
		error=criterion(output,label)
		error.backward()
		optim.step()
		print('done a batch: training error is: '+ str(error.data.cpu().numpy()))

	if(epoch%display_step==0):
		correct=0
		total=0
		for j,test_data in enumerate(test_dataloader):
			print('test entered')
			test_videos,test_labels=test_data
			if(cuda_use):
				test_videos,test_labels=Variable(test_videos.cuda()),Variable(test_labels.cuda())
			else:
				test_videos,test_labels=Variable(test_videos),Variable(test_labels)
			test_output=network(test_videos,10)
			_, predicted = torch.max(test_output.data, 1)
    		total += test_labels.size(0)
    		correct += predicted.eq(test_labels).cpu().sum()
    		print('tested a batch')
    	print('Accuracy of the network on the  test images: %d %%' % (100 * correct / total))
"""for i,data in enumerate(train_dataloader,0):
	network.zero_grad()
	video,label=data
	video,label=Variable(video),Variable(label)
	network(video,20)"""

