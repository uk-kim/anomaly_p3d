import os
import random
import PIL.Image as Image
import numpy as np

from config import *


def normalize(im):
    return im / 255.0

# def normalize(im):
#     return im * (2.0 / 255.0) - 1

# def denormalize(im):
#     return (im + 1.) / 2.

class DBreader:
    def __init__(self, dbtype='UCSDped1', batch_size=1, n_frames_clip=8, step_frames=4, resize=[224, 224], colorType=0, shuffle=True):
        '''
          Arguments:
            dbtype        : type of dataset (select one in ['UCSD', 'Avenue', 'Amit_subway'])
            batch_size    : size of batch. type: Integer, must be positive number.
            n_frames_clip : length of Sequential length,   e.g. seq_length is 3, choose t, t+1, t+2 images.
            resize        : If none, use original size. If you use certain size, resize=[h, w]
            colorType     : If image is gray space, colorType=0, else if image is color(RGB), use colorType=1.
            shuffle       : If true, the order of data files will be shuffled.
        '''
        self.dbtype        = dbtype
        self.batch_size    = batch_size
        self.n_frames_clip = n_frames_clip
        self.step_frames   = step_frames
        self.resize        = resize
        self.colorType     = colorType
        self.shuffle       = shuffle
            
        ''' Arguments Validation '''
        self.color_ch      = 3 if self.colorType else 1
        self.batch_size    = self.batch_size if self.batch_size > 0 else 1
        self.n_frames_clip = self.n_frames_clip if self.n_frames_clip > 0 else 1
        self.step_frames   = min(max(self.step_frames, 1), self.n_frames_clip)
        
        self.train_index   = 0
        self.test_index    = 0
        #self.dataset_path = ""
        
        self.trainFileList  = []
        self.testFileList   = []
        
        self.trainIndexlist = []
        self.testIndexlist  = []
        
        if self.dbtype == 'UCSDped1' or self.dbtype == 'UCSDped2':
            ext_format="tif"
            data_dir=os.path.join(UCSD_DATASET_PATH, self.dbtype)
            
            # Get list of images order by asc from train sets.
            for i, (root, _, files) in enumerate(os.walk(os.path.join(data_dir, "Train"))):
                f_list=[f for f in files if f.split('.')[-1] == ext_format]
                if len(f_list):
                    f_list.sort()
                    self.trainFileList.append([root, f_list])
            self.trainFileList.sort()
            
            # Get list of images order by asc from test sets.
            for i, (root, _, files) in enumerate(os.walk(os.path.join(data_dir, "Test"))):
                f_list=[f for f in files if f.split('.')[-1] == ext_format]
                if len(f_list):
                    f_list.sort()
                    self.testFileList.append([root, f_list])
            self.testFileList.sort()
            
            # Index mapping, dir/filename <--> dir_index/file_index
            for i, arg in enumerate(self.trainFileList):
                f_idxs=list(range(0, len(arg[1]) - self.n_frames_clip, self.step_frames))
                for idx in f_idxs:
                    self.trainIndexlist.append([i, idx])
            
            for i, arg in enumerate(self.testFileList):
                f_idxs=list(range(0, len(arg[1]) - self.n_frames_clip, self.step_frames))
                for idx in f_idxs:
                    self.testIndexlist.append([i, idx])
                    
        elif self.dbtype == 'Avenue':
            data_dir=os.path.join(config.BASE_DATASET_PATH, "Avenue_Dataset")
        else: # Maybe Amit_subway
            data_dir=os.path.join(config.BASE_DATASET_PATH, "Amit_Subway")
        
        
        if self.shuffle:
            random.shuffle(self.trainIndexlist)
            random.shuffle(self.testIndexlist)

        self.n_train_clips=len(self.trainIndexlist)
        self.n_test_clips=len(self.testIndexlist)
        
    
    def get_clip(self, idx, train_or_test=True):
        '''
          idx : if idx is default(as -1), choose image randomly, else choose image idx_th.
          train_or_test : if true, choose image from train set, else choose image from test set.
        '''
        ''' image_list : [batch_size, clip_size, height, width, channels]'''
        
        if train_or_test:
            ''' Get images clip from train data'''
            folder_idx, img_idx = self.trainIndexlist[idx]
            folder_path = self.trainFileList[folder_idx][0]

            clip_list=[]
            for f_name in self.trainFileList[folder_idx][1][img_idx:img_idx + self.n_frames_clip]:
                file_path = os.path.join(folder_path, f_name)
                img = Image.open(file_path)
                if self.resize:
                    img = img.resize(self.resize, Image.ANTIALIAS)
                img = np.array(img)
                if len(img.shape) == 2:
                    img=np.expand_dims(img, -1)
                clip_list.append(img)
        else:
            ''' Get images clip from test data'''
            folder_idx, img_idx = self.testIndexlist[idx]
            folder_path = self.testFileList[folder_idx][0]
            
            clip_list=[]
            for f_name in self.testFileList[folder_idx][1][img_idx:img_idx + self.n_frames_clip]:
                file_path = os.path.join(folder_path, f_name)
                img = Image.open(file_path)
                if self.resize:
                    img = img.resize(self.resize, Image.ANTIALIAS)
                img = np.array(img)
                if len(img.shape) == 2:
                    img=np.expand_dims(img, -1)
                clip_list.append(img)
                
        return np.array(clip_list)
    
    def next_batch(self, train_or_test=True):
        '''
          train_or_test : if true, choose image from train set, else choose image from test set.
        '''
        img_list=[]
        if train_or_test:
            ''' Get images clip from train data'''
            if self.train_index + self.batch_size > self.n_train_clips:
                if self.shuffle:
                    random.shuffle(self.trainIndexlist)
                self.train_index = 0
            
            for i in range(self.train_index, self.train_index + self.batch_size):
                clip_images = self.get_clip(i, train_or_test)
                img_list.append(clip_images)
            
            self.train_index += self.batch_size
        else:
            ''' Get images clip from test data'''
            if self.test_index + self.batch_size > self.n_test_clips:
                if self.shuffle:
                    random.shuffle(self.testIndexlist)
                self.test_index = 0
            
            for i in range(self.test_index, self.test_index + self.batch_size):
                clip_images = self.get_clip(i, train_or_test)
                img_list.append(clip_images)
            
            self.test_index += self.batch_size
        
        return np.array(img_list)
    
    def initialize(self, train_or_test=True):
        if self.shuffle:
            random.shuffle(self.trainIndexlist)
        self.train_index = 0