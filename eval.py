# To evaluate error prediction distributions

from tsc import tsc_net
from dataset import dataset
import numpy as np
import matplotlib.pyplot as plt
import copy

#def eval():
mynet = tsc_net()
mynet.restoreParam()
data = dataset()

file = 'preprocessed.pickle'

data.open_dataset(file)
train_cnt_per_class = np.zeros([43])

# Training dataset
print("Training dataset:")
X_dataset,  y_dataset  = data.load_train_data()
mynet.val(X_dataset, y_dataset, 0, False)
# Error category statistics
train_err=copy.deepcopy(mynet.err_statistics(X_dataset, y_dataset))
print('sum(train_err) = ', np.sum(train_err))
# Total training data per class
for i in range(len(y_dataset)):
    train_cnt_per_class[np.argmax(y_dataset[i])]+=1 
print("train_cnt_per_class = {}, its sum = {}".format(train_cnt_per_class, np.sum(train_cnt_per_class)))

# Validation dataset
print("Validation dataset:")
X_dataset,  y_dataset  = data.load_valid_data()
mynet.val(X_dataset, y_dataset, 0, False)
# Error category statistics
valid_err=copy.deepcopy(mynet.err_statistics(X_dataset, y_dataset))

# Test dataset
print("Test dataset:")
X_dataset,  y_dataset  = data.load_test_data()
mynet.val(X_dataset, y_dataset, 0, False)
# Error category statistics
test_err=copy.deepcopy(mynet.err_statistics(X_dataset, y_dataset))

# Err/Total ratio figure
train_err_ratio = np.divide(train_err, train_cnt_per_class)
print('train err_cnt/total_cnt = {}'.format(train_err_ratio))


# Draw the train/valid/test err count histogram figure
train_err=train_err/sum(train_err)
valid_err=valid_err/sum(valid_err)
test_err=test_err/sum(test_err)

fig, ax = plt.subplots()
rects1 = ax.bar(np.arange(len(train_err)), train_err, 0.2, color='r')
rects2 = ax.bar(np.arange(len(valid_err))+0.2, valid_err, 0.2, color='g')
rects3 = ax.bar(np.arange(len(test_err))+0.4, test_err, 0.2, color='b')
ax.legend((rects1[0], rects2[0], rects3[0]), ('train_err', 'validation_err', 'test_err'))
ax.set_xlabel('Category')
ax.set_ylabel('Percentage')
ax.set_title('Train/Validation/Test dataset per class err cnt/total')
plt.show() 

fig, ax = plt.subplots()
rects5 = ax.bar(np.arange(len(train_cnt_per_class)), train_cnt_per_class/sum(train_cnt_per_class), 0.2, color='r')
rects6 = ax.bar(np.arange(len(train_cnt_per_class))+0.2, train_err_ratio, 0.2, color='g')
ax.legend((rects5[0], rects6[0]), ('train_data_histogram', 'err_histogram'))
ax.set_xlabel('Category')
ax.set_ylabel('Percentage')
ax.set_title('Train dataset per class train_data/err cnt histogram')
plt.show() 

data.close_dataset()

#if __name__ == '__main__':
#    eval()


