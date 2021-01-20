import os
import random
import tarfile
from math import ceil, floor


from torch.utils import data
import numpy as np

from utils import image_loader, download





def create_datasetsAF(dataroot, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot)


    images_root = os.path.join(dataroot, 'African')
    names_af = os.listdir(images_root)
    if len(names_af) == 0:
        raise RuntimeError('Empty dataset')

    af_training_set =[]
    af_validation_set =[]
    for klass, name in enumerate(names_af):
        def add_class(image):
            image_path = os.path.join(images_root, name, image)
            return (image_path, klass, name)

        images_of_person = os.listdir(os.path.join(images_root, name))     
        total = len(images_of_person)
        
        af_training_set+= map(
                    add_class,
                    images_of_person[:ceil(total * train_val_split)])
        af_validation_set+= map(
                    add_class,
                    images_of_person[floor(total * train_val_split):])
     
    
    return af_training_set, af_validation_set, len(names_af)


def create_datasetsAs(dataroot, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot)


    images_root = os.path.join(dataroot, 'Asian')
    names_as = os.listdir(images_root)
    if len(names_as) == 0:
        raise RuntimeError('Empty dataset')

    as_training_set = []
    as_validation_set = []
    count=0
    for count in range(0, 1): 
        for klass, name in enumerate(names_as):
            count+=1
            def add_class(image):
                image_path = os.path.join(images_root, name, image)
                return (image_path, klass, name)

            images_of_person = os.listdir(os.path.join(images_root, name))
            total = len(images_of_person)

            as_training_set += map(
                    add_class,
                    images_of_person[:ceil(total * train_val_split)])
            as_validation_set += map(
                    add_class,
                    images_of_person[floor(total * train_val_split):])
           
            
    return as_training_set, as_validation_set, len(names_as)

def create_datasetsSA(dataroot, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot)


    images_root = os.path.join(dataroot, 'Indian')
    names_sa = os.listdir(images_root)
    if len(names_sa) == 0:
        raise RuntimeError('Empty dataset')

    sa_training_set = []
    sa_validation_set = []
    count=0
    for count in range(0, 1):
        for klass, name in enumerate(names_sa):
            count+=1
            def add_class(image):
                image_path = os.path.join(images_root, name, image)
                return (image_path, klass, name)

            images_of_person = os.listdir(os.path.join(images_root, name))
            total = len(images_of_person)

            sa_training_set += map(
                    add_class,
                    images_of_person[:ceil(total * train_val_split)])
            sa_validation_set += map(
                    add_class,
                    images_of_person[floor(total * train_val_split):])
        

    return sa_training_set, sa_validation_set, len(names_sa)

def create_datasetsW(dataroot, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot)


    images_root = os.path.join(dataroot, 'Caucasian')
    names_w = os.listdir(images_root)
    if len(names_w) == 0:
        raise RuntimeError('Empty dataset')

    w_training_set = []
    w_validation_set = []
    count=0
    for count in range(0, 1):
        count+=1
        for klass, name in enumerate(names_w):
            def add_class(image):
                image_path = os.path.join(images_root, name, image)
                return (image_path, klass, name)

            images_of_person = os.listdir(os.path.join(images_root, name))
            total = len(images_of_person)

            w_training_set += map(
                    add_class,
                    images_of_person[:ceil(total * train_val_split)])
            w_validation_set += map(
                    add_class,
                    images_of_person[floor(total * train_val_split):])
         

    return w_training_set, w_validation_set, len(names_w)

def create_datasets(dataroot, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot)

    training_set = []
    validation_set = []
    count=0
    for count in range(0, 1):
        for klass, name in enumerate(names):
            count+=1
            def add_class(image):
                image_path = os.path.join(images_root, name, image)
                return (image_path, klass, name)
            images_of_person = os.listdir(os.path.join(images_root, name))
            total = len(images_of_person)

            training_set += map(
                    add_class,
                    images_of_person[:ceil(total * train_val_split)])
            validation_set += map(
                    add_class,
                    images_of_person[floor(total * train_val_split):])

    return training_set, validation_set, len(names)




class Dataset(data.Dataset):

    def __init__(self, datasets, transform=None, target_transform=None):
        self.datasets = datasets
        self.num_classes = len(datasets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        image = image_loader(self.datasets[index][0])
        if self.transform:
            image = self.transform(image)
        return (image, self.datasets[index][1], self.datasets[index][2])


class PairedDataset(data.Dataset):

    def __init__(self, dataroot, pairs_cfg, transform=None, loader=None):
        self.dataroot = dataroot
        self.pairs_cfg = pairs_cfg
        self.transform = transform
        self.loader = loader if loader else image_loader

        self.image_names_a = []
        self.image_names_b = []
        self.matches = []
       
        self._prepare_dataset()

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, index):
        return (self.transform(self.loader(self.image_names_a[index])),
                self.transform(self.loader(self.image_names_b[index])),
                self.matches[index])
  
    def _prepare_dataset(self):
        raise NotImplementedError
        
    

class LFWPairedDataset(PairedDataset):

   def _prepare_dataset(self):
        pairs = self._read_pairs(self.pairs_cfg)
        Count_a = 1
        Count_b = 0
        for pair in pairs:
           if pairs[0][ Count_a ] == pairs[0][Count_b] :
                match = True
                index1, name1, index2, name2  = \
                    pair[0], pair[0], pair[0], pair[0]

           else:
                match = False
                index1, name1,index2, name2, = \
                    pair[0], pair[0], pair[0], pair[0]
           

           self.image_names_a.append(os.path.join(
                   self.dataroot, 'RFW-deepfunneled',
                    name1, "{}_{:04d}.jpg".format(name1, index1)))

           self.image_names_b.append(os.path.join(
                    self.dataroot, 'RFW-deepfunneled',
                    name2, "{}_{:04d}.jpg".format(name2, index2)))
           self.matches.append(match)
   def _read_pairs(self, pairs_filename):
        pairs = []
        pair_odd = []
        pair_even = []
        pair1=[]
        pair=[]
        
        with open(pairs_filename, 'r') as f:
            i = 1
            for line in f.readlines():
                if i % 2 == 0 :
                    pair_even = line.strip().split()
                    pair1.append( pair_even + pair_odd)
                else:
                    pair_odd = line.strip().split()  
                i += 1
          
        
       # for line in f.readlines():
        #    i = 1
         #   if i % 2 == 0 :
          #      pair_even = line.strip().split()
           #     pair1.append(pair_odd + pair_even)
            #else:
             #   pair_odd = line.strip().split() 
            #i += 1 
            #print(pair1)
            #pairs.append(pair1)
        #return pairs   
               # i += 1 
            print(pair1)
            pairs.append(pair1)
        return pairs  
        #    for line in f.readlines()[1:]:
        #        pair = line.strip().split()
        #        pairs.append(pair)
        #return pairs 
          
        
        #for line in f.readlines()[1:]:
                #pair = line.strip().split()
                #pair2 = line.strip().split()[::2]
                #pair1=pair+pair2
                      
        #  with open('file_name') as f:
    #for line1, line2 in zip(f, f):
     #   print(line1, line2)
        #while True:
    #line1 = f.readline()
    #line2 = f.readline()
            #with open(pairs_filename, 'r') as f:
             #   for line in f.readlines()[2:]:
            #        pair = line.strip().split()
