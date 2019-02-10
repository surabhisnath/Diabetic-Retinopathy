split_data_into_folds.py 

This file is used for data preparation, i.e splitting the data into TEST and TRAIN for Multiple FOLDS.
The outputted directory can then be fed directly into the classifier pipeline for the next step ,i.e classification.

Command line PARAMETERS:

labels_path=sys.argv[1] : Path of csv file containing file name and labels
dir_path=sys.argv[2]    : Path of directory containing the images
output=sys.argv[3] 	: The output folder of this code (Converts DATA to 5 FOLDS OF TRAIN TEST SPLITS)

