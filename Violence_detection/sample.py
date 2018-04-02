import os,glob
import skvideo.io
train_files=glob.glob("/vulcan/scratch/koutilya/violence_datasets/Hockey_dataset/train/*.avi")
test_files=glob.glob("/vulcan/scratch/koutilya/violence_datasets/Hockey_dataset/test/*.avi")

for f in train_files:
	temp=skvideo.io.vread(f)
	if(temp.shape[0]!=41):
		print(f,' ',temp.shape[0])
	else:
		continue