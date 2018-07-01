# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 10:28:03 2018

@author: A.Akl
"""

import os
import scipy.misc
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

data_path = "/emnist-digits/"
output_dir = "output_dir/"

train_images_path = os.path.join(data_path,"emnist-digits-train-images-idx3-ubyte.gz")
train_labels_path = os.path.join(data_path,"emnist-digits-train-labels-idx1-ubyte.gz")

test_images_path = os.path.join(data_path,"emnist-digits-test-images-idx3-ubyte.gz")
test_labels_path = os.path.join(data_path,"emnist-digits-test-labels-idx1-ubyte.gz")



# load training images binary file
with open(train_images_path, 'rb') as f:
  train_images = extract_images(f)
# load training labels binary file
with open(train_labels_path, 'rb') as f:
  train_labels = extract_labels(f)


# load test images binary file
with open(test_images_path, 'rb') as f:
  test_images = extract_images(f)
# load test labels binary file
with open(test_labels_path, 'rb') as f:
  test_labels = extract_labels(f)



def save_class0(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class0')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)

def save_class1(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class1')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)
        
def save_class2(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class2')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)

def save_class3(save_dir, image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class3')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)

def save_class4(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class4')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)

def save_class5(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class5')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)

def save_class6(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class6')
#    print(class_dir)
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)

def save_class7(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class7')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)

def save_class8(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class8')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)

def save_class9(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class9')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)

# dict instead of switch case or if else technique
class_label = {
        0: save_class0,
        1: save_class1,
        2: save_class2,
        3: save_class3,
        4: save_class4,
        5: save_class5,
        6: save_class6,
        7: save_class7,
        8: save_class8,
        9: save_class9
        }
  
# saving training data
i = 0
num_images = len(train_images)
for i in range(0, num_images):
        
    image = train_images[i]
    image = image.transpose([1,2,0])
    image = image.reshape(28,28)
    label = train_labels[i]
    
    class_label[label]('train',image,i) # call dict as method
    i += 1
    
 
# saving test data
i = 0
num_images = len(test_images)
for i in range(0, num_images):
        
    image = test_images[i]
    image = image.transpose([1,2,0])
    image = image.reshape(28,28)
    label = test_labels[i]
    
    class_label[label]('test',image,i) # call dict as method
    i += 1
