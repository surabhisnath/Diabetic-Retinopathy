# split_data_into_folds.py 

# This file is used for data preparation, i.e splitting the data into TEST and TRAIN for Multiple FOLDS.
# The outputted directory can then be fed directly into the classifier pipeline for the next step ,i.e classification.

# Command line PARAMETERS:

# labels_path=sys.argv[1] : Path of csv file containing file name and labels
# dir_path=sys.argv[2]    : Path of directory containing the images
# output=sys.argv[3] 	: The output folder of this code (Converts DATA to 5 FOLDS OF TRAIN TEST SPLITS)



import os
import random
import cv2
import shutil
import sys


labels_path=None

try:

	labels_path=sys.argv[1] #path of csv file containing file name and labels
	dir_path=sys.argv[2] #path of directory containing the images
	output=sys.argv[3] # the output folder of this code (Converts DATA to 5 FOLDS OF TRAIN TEST SPLITS)

except Exception as e:
	

	if labels_path==None:
		print('Default paths selected.')
	# contains labels
		labels_path='C:\\Users\\Vishal\\ML_Project\\MLBA Project\\Base1.csv'
		# directory containing images
		dir_path='C:\\Users\\Vishal\\ML_Project\\MLBA Project\\Base1'
		# create the output folder : Base1_split
		output_dir='Base1_split'


os.mkdir(output_dir)
# create a list of tuples  (file path of image,severity grade)
path_n_label=[]

with open(labels_path) as f:
		next(f)
		# parse each line 
		for line in f:
			elts=line[:-1].split(',')
			path_n_label.append((elts[0],int(elts[2])))


kfoldvalue=5


os.chdir(output_dir)
# change directory 
for i in range(kfoldvalue):

	fold_folder='Fold_'+str(i)
	print('Fold')

	os.mkdir(fold_folder)
	os.chdir(fold_folder)
	# change directory 

	random.shuffle(path_n_label)
	train_len=int(0.8*len(path_n_label))
	
	train=path_n_label[:train_len]
	test=path_n_label[train_len:]

	class_to_samples_train={}
	class_to_samples_test={}

	for pair in train:

		if pair[1] in class_to_samples_train:

			class_to_samples_train[pair[1]].append(pair[0])
		else:
			class_to_samples_train[pair[1]]=[]
			class_to_samples_train[pair[1]].append(pair[0])


	train_folder='Train'
	print('Train')

	# change directory 
	os.mkdir(train_folder)
	os.chdir(train_folder)


	for key in class_to_samples_train:
		os.mkdir(str(key))
		os.chdir(str(key))
		# change directory 

		for file_name in class_to_samples_train[key]:
			# print(file_name)
			# img = cv2.imread(dir_path+'\\'+file_name)
			source=dir_path+'\\'+file_name
			dest=os.getcwd()+'\\'+file_name[:-4]+'.tif'
			# print(os.getcwd())
			# print(os.getcwd()+'\\'+file_name[:-4]+'.tif')
			# cv2.imwrite(os.getcwd()+'\\'+file_name[:-4]+'.tif',img)
			shutil.copyfile(source,dest)

		os.chdir('..')
		# change back


	os.chdir('..')
	# change back


	test_folder='Test'

	print('Test')

	os.mkdir(test_folder)
	os.chdir(test_folder)
	# change directory 

	
	for pair in test:

		if pair[1] in class_to_samples_test:

			class_to_samples_test[pair[1]].append(pair[0])

		else:

			class_to_samples_test[pair[1]]=[]
			class_to_samples_test[pair[1]].append(pair[0])

	# print (class_to_samples)


	for key in class_to_samples_test:
		os.mkdir(str(key))
		os.chdir(str(key))
		# change directory 


		# print(class_to_samples_test[key])
		for file_name in class_to_samples_test[key]:
			source=dir_path+'\\'+file_name
			dest=os.getcwd()+'\\'+file_name[:-4]+'.tif'
			# print(os.getcwd())
			# print(os.getcwd()+'\\'+file_name[:-4]+'.tif')
			# cv2.imwrite(os.getcwd()+'\\'+file_name[:-4]+'.tif',img)
			shutil.copyfile(source,dest)
		os.chdir('..')
		# change back


	os.chdir('..')
	# change back

	os.chdir('..')
	# change back






exit()






