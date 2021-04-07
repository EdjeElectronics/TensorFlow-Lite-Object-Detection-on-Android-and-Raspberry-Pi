import os
import argparse
import shutil
from tqdm import tqdm


PATH_TRUTH = './input/ground-truth/'
PATH_IMAGES = './input/images-optional/'

def load_negatives(path):
    path_neg = path + '/neg/'
    images = os.listdir(path_neg)
    
    print('Copying negatives images...')
    for image in tqdm(images):
        shutil.copyfile(path_neg + '/' + image, PATH_IMAGES + '/' + image)
        title_txt = image.split('.')[0]
        
        file_dest = open(PATH_TRUTH + '/' + title_txt + '.txt', 'w')
        file_dest.close()
            

def load_positives(path):
    path_pos = path + '/pos/'
    path_detections = path + '/annotations/'
    
    files = os.listdir(path_detections)
    images = os.listdir(path_pos)
    
    print('Copying positives images...')
    for image in tqdm(images):
        shutil.copyfile(path_pos + image, PATH_IMAGES + '/' + image)
        
    print('Creating detections files of positives images...')    
    for file in tqdm(files):
        with open(PATH_TRUTH + '/' + file, 'a') as file_dest:
            with open(path_detections + file, 'r', encoding = "ISO-8859-1") as file_src:
                
                for line in file_src.readlines():
                    
                    if 'Bounding' in line:
                        label = line.split('"')[1].strip()
                        label = 'person' if label == 'PASperson' else label
                        
                        fila_detecciones = line.split(':')[1].strip()
                        
                        xmin = int(fila_detecciones
                                   .split('(')[1]
                                   .split(',')[0])
                        
                        ymin = int(fila_detecciones
                                   .split('(')[1]
                                   .split(',')[1]
                                   .split(')')[0])
                        
                        xmax = int(fila_detecciones
                                   .split('(')[2]
                                   .split(',')[0])
                        
                        ymax = int(fila_detecciones
                                   .split('(')[2]
                                   .split(',')[1]
                                   .split(')')[0])
                        
                        new_line = '{} {} {} {} {}\n'.format(label,
                                                           xmin,
                                                           ymin,
                                                           xmax,
                                                           ymax)
                        file_dest.write(new_line)
                        
    
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Folder of dataset', required=True)
    args = parser.parse_args()
    path = './' + args.path
    dirs_input = ['detection-results',
                  'ground-truth',
                  'images-optional']
    
    for dir_input in dirs_input: 
        if not os.path.exists('./input/' + dir_input):
            os.makedirs('./input/' + dir_input)
    load_negatives(path)
    load_positives(path)
    