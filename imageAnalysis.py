import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('trainset.csv')
dataset_size = len(dataset)

learn_dataset = dataset[0:int(0.8*dataset_size)]
validation_dataset = dataset[int(0.8*dataset_size):dataset_size]

learn_labels = learn_dataset['label']
learn_imgs = learn_dataset.iloc[:,1:785]
learn_imgs = learn_imgs / 255