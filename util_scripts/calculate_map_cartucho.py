##### Calculate object detection mAP score #####
#
# Author: Evan Juras, EJ Technology Consultants (https://ejtech.io)
# Date: 11/12/22
#
# Description: 
# This script determines an object detection model's mAP score using a calculator from https://github.com/Cartucho/mAP .
# It calculates the COCO metric (mAP @ 0.5:0.95) by running the calculator script ("main.py") multiple times at different
# IoU thresholds, then averaging the result from each run. It also supports Pascal VOC metric (mAP @ 0.5) and custom user metrics.

import os
import sys
import argparse
import numpy as np

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--labels', help='Path to the labelmap file', default='labelmap.txt')
parser.add_argument('--outdir', help='Output folder to save results in', default='outputs')
parser.add_argument('--metric', help='mAP metric to calculate: "coco", "pascalvoc", or "custom"', default='coco')
parser.add_argument('--iou', help='(Only if using --metric=custom) Specify IoU threshholds \
    to use for evaluation (example: 0.5,0.6,0.7)')
parser.add_argument('--show_images', help='Display and save images as they are evaluated', action='store_true') # Coming soon!
parser.add_argument('--show_plots', help='Display and save plots showing precision/recall curve, mAP score, etc', action='store_true') # Coming soon!

args = parser.parse_args()

labelmap = args.labels
outputs_dir = args.outdir
metric = args.metric
show_imgs = args.show_images
show_plots = args.show_plots

# Define which metric to use (i.e. which set of IoU thresholds to calculate mAP for)
if metric=='coco':
    iou_threshes = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
elif metric=='pascalvoc':
    iou_threshes = [0.5]
elif metric=='custom':
    custom_ious = args.iou
    try:
        iou_threshes = [float(iou) for iou in custom_ious]
    except:
        print('Invalid entry for --iou. Example of a correct entry: "--iou=0.5,0.6,0.7"')
        sys.exit()
else:
    print('Invalid entry for --metric. Please use coco, pascalvoc, or custom.')
    sys.exit()

# Get file paths
cwd = os.getcwd()
output_path = os.path.join(cwd,outputs_dir)
labelmap_path = os.path.join(cwd,labelmap)

# Define arguments to show images and plots (if desired by user)
if show_imgs: show_img_arg = ''
else: show_img_arg = ' -na' # "-na" argument tells main.py NOT to show images

if show_plots: show_plot_arg = ''
else: show_plot_arg = ' -np' # "-np" argument tells main.py NOT to show plots


# Load the label map
with open(labelmap_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Make folder to store output result files
if os.path.exists(output_path):
    print('The output folder %s already exists. Please delete it or specify a different folder name using --outdir.' % output_path)
    sys.exit()
else:
    os.makedirs(output_path)

# Create dictionary to store overall mAP results and results for each class
mAP_results = {'overall':np.zeros(len(iou_threshes))}
for classname in classes:
    mAP_results[classname] = np.zeros(len(iou_threshes)) # Add each class to dict

for i, iou_thresh in enumerate(iou_threshes):

    # Modify main.py to use the specified IoU value
    with open('main.py', 'r') as f:
        data = f.read()
        
        # Set IoU threshold value
        data = data.replace('MINOVERLAP = 0.5', 'MINOVERLAP = %.2f' % iou_thresh)
        f.close()

    with open('main_modified.py', 'w') as f:
        f.write(data)

    # Run modified script
    print('Calculating mAP at %.2f IoU threshold...' % iou_thresh)
    os.system('python main_modified.py' + show_img_arg + show_plot_arg)

    # Extract mAP values by manually parsing the output.txt file
    with open('output/output.txt', 'r',) as f:
        for line in f:
            if '%' in line:
                # Overall mAP result is stored as "mAP = score%" (example: "mAP = 63.52%")
                if 'mAP' in line:
                    vals = line.split(' ')
                    overall_mAP = float(vals[2].replace('%',''))
                    mAP_results['overall'][i] = overall_mAP
                # Class mAP results are stored as "score% = class AP" (example: "78.30% = dime AP")
                else:
                    vals = line.split(' ')
                    class_name = vals[2]
                    class_mAP = float(vals[0].replace('%',''))
                    mAP_results[class_name][i] = class_mAP

    # Save mAP results for this IoU value as a different folder name, then delete modified script
    newpath = os.path.join(output_path,'output_iou_%.2f' % iou_thresh)
    os.rename('output',newpath)
    os.remove('main_modified.py')

# Okay, we found mAP at each IoU value! Now we just need to average the mAPs and display them.
class_mAP_result = []
print('\n***mAP Results***\n')
print('Class\t\tAverage mAP @ 0.5:0.95')
print('---------------------------------------')
for classname in classes:
    class_vals = mAP_results[classname]
    class_avg = np.mean(class_vals)
    class_mAP_result.append(class_avg)
    print('%s\t\t%0.2f%%' % (classname, class_avg)) # TO DO: Find a better variable name than "classname"

overall_mAP_result = np.mean(class_mAP_result)
print('\nOverall\t\t%0.2f%%' % overall_mAP_result)
        
