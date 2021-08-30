import gzip
import hashlib
import os
import pickle
import sys
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np


def get_one_hot(targets, nb_classes):                                              
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]    


def show_progress(blk_num, blk_sz, tot_sz):                                        
    percentage = 100. * blk_num * blk_sz / tot_sz                                  
    print("Progress: %.1f %%" % percentage, end="\r", flush=True)
    
    
def md5_checksum(file_path):                                                       
    with open(file_path, "rb") as f:                                               
        checksum = hashlib.md5(f.read()).hexdigest()                               
    return checksum


def download_url(url, file_path, checksum):                                        
    # create directory if needed                                                   
    file_dir = os.path.dirname(file_path)                                          
    if not os.path.exists(file_dir):                                               
        os.makedirs(file_dir)                                                      
                                                                                   
    if os.path.exists(file_path):                                                  
        # check md5                                                                
        if md5_checksum(file_path) == checksum:                                    
            print("{} already exists.".format(file_path))                          
            return                                                                 
        else:                                                                      
            print("Wrong checksum!")                                               
                                                                                   
    try:                                                                           
        print("Downloading {} to {}".format(url, file_path))                       
        urlretrieve(url, file_path, show_progress)                                 
    except URLError:                                                               
        raise RuntimeError("Error downloading resource!")                          
    except KeyboardInterrupt:                                                      
        print("Interrupted")                  
        
        
def mnist(data_dir, one_hot=False):                                                
    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"                        
    checksum = "a02cd19f81d51c426d7ca14024243ce9"                                  
                                                                                   
    save_path = os.path.join(data_dir, url.split("/")[-1])                         
    print("Preparing MNIST dataset ...")                                           
    try:                                                                           
        download_url(url, save_path, checksum)                                     
    except Exception as e:                                                         
        print("Error downloading dataset: %s" % str(e))                            
        sys.exit(1)                                                                
                                                                                   
    # load the dataset                                                             
    with gzip.open(save_path, "rb") as f:                                          
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")         
                                                                                   
    if one_hot:                                                                    
        train_set = (train_set[0], get_one_hot(train_set[1], 10))                  
        valid_set = (valid_set[0], get_one_hot(valid_set[1], 10))                  
        test_set = (test_set[0], get_one_hot(test_set[1], 10))                     
                                                                                   
    return train_set, valid_set, test_set      


def make_dataset():
    train_set, _, test_set = mnist("./data")
    test_x, test_y = test_set
    full_train_x, full_train_y = train_set

    train_idx = []
    for label in range(10):
        idx_list = np.where(full_train_y == label)[0]
        train_idx.append(np.random.choice(idx_list))
    train_x, train_y = full_train_x[train_idx], full_train_y[train_idx]
    print("train sample idx: ", train_idx)

    return train_x, train_y, test_x, test_y
