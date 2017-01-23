#
# Predict class of traffic sign images from internet
#
from tsc import tsc_net
from dataset import dataset
import numpy as np
import cv2
import tensorflow as tf

def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    img_max = np.max(image_data)
    img_min = np.min(image_data)
    a = 0.1
    b = 0.9

    img_normed = a + (b-a)*(image_data - img_min)/(img_max - img_min)
    #print(np.max(img_normed))
    #print(np.min(img_normed))
    return img_normed

def normalize_color(image_data):
    """
    Normalize the image data on per channel basis. 
    """
    img_normed_color = np.zeros_like(image_data, dtype=float)
    for ch in range(image_data.shape[3]):
        tmp = normalize_grayscale(image_data[:,:,:,ch])
        img_normed_color[:,:,:,ch] = tmp
    #print(np.max(img_normed_color))
    #print(np.min(img_normed_color))
    return img_normed_color

# Create TSC NN and restore trained parameters
mynet=tsc_net()
mynet.restoreParam()
data = dataset()
data.open_dataset('preprocessed.pickle')

# Load several samples from training dataset to sanitize the process
X, y, _ = data.load_train_batch(10)
X_in = X[0].reshape(-1,32,32,3)
mynet.predict(X_in)
X_in = X[9].reshape(-1,32,32,3)
mynet.predict(X_in)
mynet.predict(X)
mynet.err_statistics(X,y)

# Load internet images and make prediction
img_in = np.zeros_like(X)
for i in range(10):
    img=cv2.imread('./internet_traffic_signs/'+str(i)+'.png')
    img = cv2.resize(img,(32,32))
    img = img.reshape(-1,32,32,3)
    # Normalize the image between [0.1, 0.9]
    img_in[i] = normalize_color(img)

# Predict
certainty=mynet.predict(img_in)

sess = tf.Session()
sess.run(tf.nn.top_k(certainty[0], k=5))

