# ### 4. Stitching everthing together

from tsc import tsc_net
from dataset import dataset
import numpy as np

EPOCH = 100
BATCH_SZ = 256
LEARNING_RATE = 8e-4
DECAY_LR = 1.0


def main():
    mynet = tsc_net()
    data = dataset()

    file = 'preprocessed.pickle'

    data.open_dataset(file)

    # Training
    lr = LEARNING_RATE
    for j in range(EPOCH):
        steps_per_epoch = data.steps_per_epoch(BATCH_SZ)
        print("Epoch {0:4d}, Steps per epoch {1:5d}".format(j, steps_per_epoch))
        for i in range(steps_per_epoch):
            X_train_batch, y_train_batch, end_of_train_dataset = data.load_train_batch(BATCH_SZ)
            if i==0 and j==0 :
                print("X_train_batch.shape = ", np.shape(X_train_batch))
                print("X_train_batch.dtype = ", X_train_batch.dtype)
                print("y_train_batch.shape = ", np.shape(y_train_batch))
                print("y_train_batch.dtype = ", y_train_batch.dtype)
            if i%500 == 0:
                    lr = lr*DECAY_LR
            mynet.train(X_train_batch, y_train_batch, lr, j*steps_per_epoch+i)

        # Validation dataset
        X_valid_dataset, y_valid_dataset = data.load_valid_data()
        mynet.val(X_valid_dataset, y_valid_dataset, j*steps_per_epoch+i, True)

        # Reset pointers test
        print("Testing dataset.reset_ptr()...")
        data.reset_ptr()

    # Test dataset
    X_test_dataset,  y_test_dataset  = data.load_test_data()
    mynet.val(X_test_dataset, y_test_dataset, j*steps_per_epoch+i, True)

    # Error category statistics
    mynet.err_statistics(X_test_dataset, y_test_dataset)

    data.close_dataset()

    mynet.saveParam()

if __name__ == '__main__':
    main()


