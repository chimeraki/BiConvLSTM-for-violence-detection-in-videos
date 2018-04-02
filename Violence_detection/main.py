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
train_dataloader=DataLoader(train_dataset,batch_size=2,shuffle=True)

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

for epoch in range(1,epochs+1):
	for i,data in enumerate(train_dataloader,0):
		network.zero_grad()
		video,label=data
                #split 20 frames of video up into 80-20%
                video_future=video[:,:,8:,:,:]
                video=video[:,:,:8,:,:]
		if(cuda_use):
			video,label,video_future=Variable(video.cuda()),Variable(label.cuda()),Variable(video_future.cuda())
		else:
			video,label,video_future=Variable(video),Variable(label),Variable(video_future)
		optim.zero_grad()
                rev_video = flip_var(video, 0)
		output,past,future=network(video,10)
		#print(output.shape)
		error=criterion(output,label)+criterion(rev_video,past)+criterion(video_future,future)   #loss is sum of future prediction, past prediction and label prediction
		error.backward()
		optim.step()

	if(epoch%display_step==0):
		print('Epoch: '+str(epoch)+' Training error: '+ str(error.data.cpu().numpy()))
		correct=0
		total=0
		for j,test_data in enumerate(test_dataloader):
			test_videos,test_labels=test_data
			if(cuda_use):
				test_videos,test_labels=Variable(test_videos.cuda()),Variable(test_labels.cuda())
			else:
				test_videos,test_labels=Variable(test_videos),Variable(test_labels)
			test_output=network(test_videos,10)
			_, predicted = torch.max(test_output.data, 1)
			total += test_labels.size(0)
    			correct += predicted.eq(test_labels.data).cpu().sum()
    		print('Accuracy of the network on the  test images: %d / %d : %d %%' % (correct,total,100 * correct / total))
"""for i,data in enumerate(train_dataloader,0):
	network.zero_grad()
	video,label=data
	video,label=Variable(video),Variable(label)
	network(video,20)"""

