import skvideo.io
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os,glob
from PIL import Image
import torch

class RandomCrop(object):
    """Crop randomly the frames in a clip.
	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip):

        h, w = clip.size()[2:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        clip = clip[:, :, top: top + new_h,
               left: left + new_w]

        return clip

class VideoDataset(Dataset):
	def __init__(self,root_dir='/vulcan/scratch/koutilya/violence_datasets/Hockey_dataset/',transform=None,mean=[128,128,128]):
		super(VideoDataset,self).__init__()

		self.root_dir=root_dir
		self.data_files=glob.glob(self.root_dir+'*.avi')
		self.file_names=os.listdir(self.root_dir)
		self.gt=[]
		self.mean=mean
		self.transform=transform

		for f in self.file_names:
			if 'fi' in f:
				self.gt.append(1)
			else:
				self.gt.append(0)
		self.gt_labels=torch.from_numpy(np.asarray(self.gt)).long()
	def __len__(self):
		return len(self.gt)

	def __getitem__(self,idx):
		videodata=skvideo.io.vread(self.data_files[idx])
		videodata=videodata[:40,:,:,:]
		videodata=videodata[::2,:,:,:] #consider equally spaced 20 frames for efficient computation time
		self.frame_count,self.height,self.width,self.channels=videodata.shape[0],videodata.shape[1],videodata.shape[2],videodata.shape[3]

		vid_data=np.zeros((self.channels,self.frame_count,224,224))
		for i in range(self.channels):
			videodata[:,:,:,i]-=self.mean[i]
		videodata/=255

		for i in range(self.frame_count):
			temp_img=Image.fromarray(videodata[i,:,:,:])
			temp_array=np.array(temp_img.resize((224,224))).transpose((2,0,1))
			#print(temp_array.shape)
			vid_data[:,i,:,:]=temp_array

		#videodata=videodata.transpose((3,0,2,1))
		vid_data=torch.from_numpy(vid_data)
		vid_data=vid_data.type(torch.FloatTensor)
		if self.transform:
			vid_data=self.transform(vid_data)

		return (vid_data,self.gt[idx])#self.gt_labels[idx,:])