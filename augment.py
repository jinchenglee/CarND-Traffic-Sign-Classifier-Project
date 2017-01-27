import pickle

training_file = "/home/vitob/git_repositories/CarND-Traffic-Sign-Classifier-Project/train.p"
testing_file = "/home/vitob/git_repositories/CarND-Traffic-Sign-Classifier-Project/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# ------------------------------------
# Dataset Summary & Exploration
# ------------------------------------
import numpy as np

### Replace each question mark with the appropriate value.

n_train = np.shape(X_train)[0]
n_test = len(X_test)
image_shape = np.shape(X_train)[1:4]
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


#<<JC>> # ------------------------------------
#<<JC>> # Visualize the German Traffic Signs Dataset histogram
#<<JC>> # ------------------------------------
#<<JC>> 
#<<JC>> ### Data exploration visualization goes here.
#<<JC>> ### Feel free to use as many code cells as needed.
#<<JC>> import matplotlib.pyplot as plt
#<<JC>> import random
#<<JC>> 
#<<JC>> # Visualizations will be shown in the notebook.
#<<JC>> get_ipython().magic('matplotlib inline')
#<<JC>> 
#<<JC>> # Show histograms of all classes in training dataset
#<<JC>> plt.hist(y_train,range(n_classes+1))
#<<JC>> plt.hist(y_test,range(n_classes+1))
#<<JC>> plt.title("Histogram of training/test classes")
#<<JC>> plt.show()
#<<JC>> 
#<<JC>> # Show a random of n example of each class
#<<JC>> samples_per_class = 2
#<<JC>> 
#<<JC>> plt.figure(figsize = (6,32))
#<<JC>> print('Snapshot of samples from training dataset:')
#<<JC>> for i in range(n_classes):
#<<JC>>     #print(i)
#<<JC>>     idxs = np.random.choice(np.flatnonzero(y_train==i), samples_per_class)
#<<JC>>     for j , idx in enumerate(idxs):
#<<JC>>         plt_idx = i*samples_per_class + j + 1
#<<JC>>         plt.subplot(43, samples_per_class, plt_idx)
#<<JC>>         plt.imshow(X_train[idx].astype('uint8'))
#<<JC>>         plt.axis('off')
#<<JC>>         
#<<JC>> plt.show()
    
    
# --------------------------------
# Data augmentation:
# --------------------------------
# Create more data for general application
# - Wrap images
# - Rotation 
# TODO:
# - Add noises to images
# - Image occlusion
# - Brightness/Contrast 
# - Scale of the traffic sign, but to a limit when shrinking.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm

import os
from sklearn.utils import shuffle
from rotate_n_warp import *

X_aug_r = np.zeros_like(X_train)
y_aug_r = np.copy(y_train)
X_aug_w = np.zeros_like(X_train)
y_aug_w = np.copy(y_train)

for i in range(n_train):
#for i in range(10):
    #image = cv2.imread("internet_traffic_signs/1.png")
    image = X_train[i]

    #cv2.imshow("Original Image", image)

    # random generate the rotate angle
    angle = random.randrange(-20,20)
    X_aug_r[i] = rotate_image(image, angle)
    X_aug_w[i] = warp_image(image)
    
    #cv2.imwrite("rotated"+str(i)+".png", X_aug_r[i])
    #cv2.imwrite("warpped"+str(i)+".png", X_aug_w[i])

print("Data augmentation Done.")

n_aug = np.shape(X_aug_r)[0]
n_test = len(y_aug_r)
assert n_aug == n_test
image_shape = np.shape(X_aug_r)[1:4]
n_classes = len(np.unique(y_aug_r))
print("Number of augment-rotated images:", n_aug)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

n_aug = np.shape(X_aug_w)[0]
n_test = len(y_aug_w)
assert n_aug == n_test
image_shape = np.shape(X_aug_w)[1:4]
n_classes = len(np.unique(y_aug_w))
print("Number of augment-warpped images:", n_aug)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# ------------------------------------
# Visualize the German Traffic Signs Dataset histogram
# ------------------------------------

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random

# Show histograms of all classes in training dataset
plt.hist(y_aug_r,bins=range(n_classes+1))
plt.hist(y_test,range(n_classes+1))
plt.title("Histogram of aug-rotated training/test classes")
plt.show()
 
# Merge augmented data into training set
X_train = np.vstack((X_train, X_aug_r))
X_train = np.vstack((X_train, X_aug_w))
y_train = np.hstack((y_train, y_aug_r))
y_train = np.hstack((y_train, y_aug_w))

n_train = np.shape(X_train)[0]
n_test = len(X_test)
image_shape = np.shape(X_train)[1:4]
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# Show a random of n example of each class
samples_per_class = 5

plt.figure(figsize = (6,32))
print('Snapshot of samples from training dataset:')
for i in range(n_classes):
    #print(i)
    idxs = np.random.choice(np.flatnonzero(y_train==i), samples_per_class)
    for j , idx in enumerate(idxs):
        plt_idx = i*samples_per_class + j + 1
        plt.subplot(43, samples_per_class, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        
plt.savefig('snapshot_train_dataset.png')
plt.show()

#---------------------------
# Save into pickle file
#---------------------------

# Shuffle training data
print("shuffle X_train, size =", X_train.shape, y_train.shape)
X_train_aug, y_train_aug = shuffle(X_train, y_train)
print("shuffle X_train_aug, size =", X_train_aug.shape, y_train_aug.shape)

# Image normalization
X_train_p = normalize_color(X_train_aug)
X_test_p = normalize_color(X_test)

# Labels binarization encoding
encoder = LabelBinarizer()
encoder.fit(y_train_aug)
y_train_p = encoder.transform(y_train_aug)
y_test_p  = encoder.transform(y_test)

# Debug print - sanity check
#print("X_test_p.shape", X_test_p.shape)
#print("y_test_p.shape", y_test_p.shape)
#print("Before test splitting, X_train sample:", X_train_p[1])
#print("Before test splitting, X_test sample:", X_test_p[1])
#print("Before test splitting, np.max(X_train_aug): ", np.max(X_train_p))
#print("Before test splitting, np.min(X_train_aug): ", np.min(X_train_p))
#print("Before test splitting, np.max(X_test): ", np.max(X_test_p))
#print("Before test splitting, np.min(X_test): ", np.min(X_test_p))
#print("Before test splitting, y_train_aug sample:", y_train_p[12])
#print("Before test splitting, y_test sample:", y_test_p[12])


# Split half test samples to be validation samples. - Reshuffle included already.
#Thus train/validation/test = 75%/12.5%/12.5%
X_valid_p, X_test_p, y_valid_p, y_test_p = train_test_split(
    X_test_p,
    y_test_p,
    test_size=0.5,
    random_state = 66)

# Debug print - sanity check
#print("After test splitting, X_train_aug sample:", X_train_p[1])
#print("After test splitting, X_valid sample:", X_valid_p[1])
#print("After test splitting, X_test sample:", X_test_p[1])
#print("After test splitting, y_train_aug sample:", y_train_p[12])
#print("After test splitting, y_valid sample:", y_valid_p[12])
#print("After test splitting, y_test sample:", y_test_p[12])
print("After test splitting, np.max(X_train_aug): ", np.max(X_train_p))
print("After test splitting, np.min(X_train_aug): ", np.min(X_train_p))
print("After test splitting, np.mean(X_train_aug): ", np.mean(X_train_p))
print("After test splitting, X_train_aug.dtype: ", X_train_p.dtype)

print("After test splitting, np.max(X_valid): ", np.max(X_valid_p))
print("After test splitting, np.min(X_valid): ", np.min(X_valid_p))
print("After test splitting, np.mean(X_valid): ", np.mean(X_valid_p))
print("After test splitting, X_valid.dtype: ", X_valid_p.dtype)

print("After test splitting, np.max(X_test): ", np.max(X_test_p))
print("After test splitting, np.min(X_test): ", np.min(X_test_p))
print("After test splitting, np.mean(X_test): ", np.mean(X_test_p))
print("After test splitting, X_test.dtype: ", X_test_p.dtype)

# Save the data for easy access
pickle_file = 'augmented.pickle'
if not os.path.isfile(pickle_file):
    print('Saving augmented preprocessed data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': X_train_p,
                    'train_labels': y_train_p,
                    'valid_dataset': X_valid_p,
                    'valid_labels': y_valid_p,
                    'test_dataset': X_test_p,
                    'test_labels': y_test_p,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')


