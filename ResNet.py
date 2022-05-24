#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists, expanduser
from keras.preprocessing import image
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from keras.applications.imagenet_utils import decode_predictions


# In[29]:


from keras.preprocessing.image import ImageDataGenerator


# In[3]:


all_xray_df = pd.read_csv('C:/Users/gayat/Downloads/archive/sample/sample_labels.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('C:', 'Users','gayat','Downloads','archive','sample' , 'images*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers:', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)


# In[4]:


img_folder = 'C:/Users/gayat/Downloads/archive/sample'
from imutils import paths
all_image_paths =  {os.path.basename(x):x for x in list(paths.list_images('C:/Users/gayat/Downloads/archive/sample/images'))}
print('Scans found:', len(all_image_paths), ', Total Headers:', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)


# In[5]:


label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)


# In[6]:


all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
from itertools import chain
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df.sample(3)


# In[7]:


print('All Labels ({})'.format(len(all_labels)), 
      [(c_label,int(all_xray_df[c_label].sum())) for c_label in all_labels])


# In[8]:


sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
print(sample_weights)
sample_weights /= sample_weights.sum()
all_xray_df = all_xray_df.sample(5000, weights=sample_weights)

label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)


# In[9]:


label_counts


# In[10]:


label_counts = 100*np.mean(all_xray_df[all_labels].values,0)


# In[11]:


label_counts


# In[12]:


label_counts = 100*np.mean(all_xray_df[all_labels].values,0)
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
ax1.set_xticklabels(all_labels, rotation = 90)
ax1.set_title('Adjusted Frequency of Diseases in Patient Group')
_ = ax1.set_ylabel('Frequency (%)')


# In[13]:


all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])


# In[14]:


all_xray_df


# In[16]:


from sklearn.model_selection import train_test_split
train_df, valid_df1 = train_test_split(all_xray_df, 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = all_xray_df['Finding Labels'].map(lambda x: x[:4]))
print('train', train_df.shape[0], 'validation', valid_df1.shape[0])
print(all_labels)


# In[18]:


valid_df, test_df = train_test_split(valid_df1, 
                                   test_size = 0.5, 
                                   random_state = 2018,
                                   stratify = valid_df1['Finding Labels'].map(lambda x: x[:4]))


# In[19]:


print('valid', valid_df.shape[0], 'test', test_df.shape[0])
print(all_labels)


# In[23]:


train_df.head()


# In[20]:


def check_for_leakage(df1, df2, patient_col):
    df1_patients_unique = set(df1[patient_col].values)
    df2_patients_unique = set(df2[patient_col].values)
    
    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    # leakage contains true if there is patient overlap, otherwise false.
    leakage = len(patients_in_both_groups) > 0 # boolean (true if there is at least 1 patient in both groups)
    return leakage


# In[24]:


print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'Patient ID')))
print("leakage between valid and test: {}".format(check_for_leakage(valid_df, test_df, 'Patient ID')))


# In[25]:


def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 224, target_h = 224):
        
    print("getting train generator...") 
    # normalizing images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator


# In[27]:


def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 224, target_h = 224):
   
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="Image Index", 
        y_col=all_labels, 
        class_mode="raw", 
        batch_size=sample_size,
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator


# In[30]:


IMAGE_DIR = "C:/Users/gayat/Downloads/archive/sample/images"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image Index", all_labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image Index", all_labels)


# In[31]:


def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
  
    
    # total number of patients (rows)
    N =labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

   
    return positive_frequencies, negative_frequencies


# In[32]:


freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos


# In[35]:


import seaborn as sns
data = pd.DataFrame({"Class": all_labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": all_labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)


# In[36]:


pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights


# In[37]:


data = pd.DataFrame({"Class": all_labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": all_labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data);


# In[38]:


test_X, test_Y = next(test_generator)


# In[40]:


t_x, t_y = next(train_generator)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -1.5, vmax = 1.5)
    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(all_labels, c_y) 
                             if n_score>0.5]))
    c_ax.axis('off')


# In[42]:


from __future__ import print_function, division
from builtins import range, input

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet import ResNet50, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="Downloads\model.weights.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=3)
callbacks_list = [checkpoint, early]
class_weights = {0: 1.0787931408725853,
 1: 5.166320166320166,
 2: 3.7298311444652907,
 3: 6.675621222296844,
 4: 1.4675919090506422,
 5: 7.161383285302594,
 6: 10.22633744855967,
 7: 79.52,
 8: 1.0,
 9: 3.4694589877835953,
 10: 3.202319587628866,
 11: 9.1109074243813,
 12: 38.23076923076923,
 13: 5.6477272727272725}


res = ResNet50(input_shape=t_x.shape[1:], weights='imagenet', include_top=False)
# don't train existing weights
for layer in res.layers:
  layer.trainable = False


# In[43]:


from keras.layers import Dropout,AveragePooling2D


# In[44]:


x = AveragePooling2D(pool_size = (4,4))(res.output)


# In[45]:


x = Flatten()(x)
x = Dense(64,activation='relu')(x)
x= Dropout(0.5)(x)
prediction = Dense(len(all_labels),activation='softmax')(x)


# In[46]:


model= Model(inputs=res.input,outputs=prediction)
model.summary()


# In[51]:


model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[52]:


r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=10,
  steps_per_epoch= 3750/32,
  callbacks = callbacks_list,
  validation_steps= 625/ 256,
  class_weight = class_weights
)


# In[ ]:




