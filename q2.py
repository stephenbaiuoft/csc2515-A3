import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)


class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch


class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0


    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters

        self.vel = - self.lr * grad + self.beta * self.vel
        return params + self.vel


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        # c is the regularization parameter by definition
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)


    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # max{1 - y_i * (w * x_i), 0 }

        # Implement hinge loss
        loss = 1-y*(self.w * X)
        # set indices less than 0 as 0
        loss[np.where(loss < 0)] = 0
        return loss


    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        # assume X, y are batched inputs
        k = X.shape[0]
        hinge_loss_vector = self.hinge_loss(X, y)
        indice = hinge_loss_vector > 0

        # step 1.0: compute sub-gradient
        gradient_t = self.c * self.w[indice] \
                     - np.sum(y[indice] * X[indice]) / k

        return gradient_t


    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        return None


def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets


def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''

    def func_grad(x):
        return 0.02 * x

    def get(x):
        return 0.01 * x * x

    w = w_init
    w_history = [w_init]
    for _ in range(steps):
        # Optimize and update the history
        # compute gradient and momentum here

        # now optimize, and update w
        # note: followed the demo and not gonna perform get(w) scale
        w = optimizer.update_params(w, func_grad(w))
        w_history.append(w)

    print(w_history)
    return w_history


def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''

    # add columns of ones to train_data
    train_data = add_bias(train_data)
    feature_count = train_data[0]

    batchsampler = BatchSampler(train_data, train_targets, batchsize)

    # init svm object: feature for bias
    # make sure penalty is > 0?!!
    svm = SVM(penalty, feature_count)
    for t in range(1, iters + 1):
        # compute learning_rate
        learning_rate = 1 / (t * penalty)

        # get batch data
        X_batch, y_batch = batchsampler.get_batch()
        sub_grad = svm.grad(X_batch, y_batch)

        # update the svm.w (params, gradient)
        svm.w = optimizer.update_params(svm.w, sub_grad)

        # optimize based on svm?
        w_t_half = svm.w - learning_rate * sub_grad
        # now update the next parameter
        ratio = 1/((w_t_half ** 2) * np.sqrt(penalty))
        svm.w = w_t_half * np.min([1, ratio])

    return svm


# add columns of ones to X_data
def add_bias(X_data):
    # add columns of ones to train_data
    n = X_data.shape[0]
    # debug
    print("original shape is: ", X_data.shape)
    train_data = np.append(X_data, np.ones(shape=(n,1)), axis=1)
    print("modified shape is now: ", X_data.shape)


def part2_3():
    # prints the required parameters
    def evaluate(svm, train_data, train_targets,
                 test_data, test_targets):
        # 1. The training loss: the hinge loss.
        hinge_loss_vector = svm.hinge_loss(train_data, train_targets)
        hinge_loss = np.sum(hinge_loss_vector) / train_data.shape[0]
        print("Train: the averaged hinge_loss is: ", hinge_loss)
        # 2. The test loss.
        hinge_loss_vector = svm.hinge_loss(test_data, test_targets)
        hinge_loss = np.sum(hinge_loss_vector) / test_data.shape[0]
        print("Test: the averaged hinge_loss is: ", hinge_loss)

        # 3. The classification accuracy on the training set.
        predict_a = (svm.w * train_data) > 0
        predict_b = (svm.w * train_data) < 0

        a_i_4 = train_targets[predict_a] == 4
        a_i_9 = train_targets[predict_a] == 9

        b_i_4 = train_targets[predict_b] == 4
        b_i_9 = train_targets[predict_b] == 9

        correct_count = 0
        # check which predict_a is for class 4 majority
        if train_targets[a_i_4].shape[0] > \
            train_targets[b_i_4].shape[0]:
            correct_count =\
                train_targets[a_i_4].shape[0] + train_targets[b_i_9].shape[0]
        # predict_a is for class 9 majority
        else:
            correct_count =\
                train_targets[a_i_9].shape[0] + train_targets[b_i_4].shape[0]

        train_accuracy = correct_count / train_data.shape[0]
        print("Training accuracy is: ", train_accuracy)

        # 4. The classification accuracy on the test set.
        predict_a = (svm.w * train_data) > 0
        predict_b = (svm.w * train_data) < 0

        a_i_4 = test_targets[predict_a] == 4
        a_i_9 = test_targets[predict_a] == 9

        b_i_4 = test_targets[predict_b] == 4
        b_i_9 = test_targets[predict_b] == 9

        correct_count = 0
        # check which predict_a is for class 4 majority
        if test_targets[a_i_4].shape[0] > \
                test_targets[b_i_4].shape[0]:
            correct_count = \
                test_targets[a_i_4].shape[0] + test_targets[b_i_9].shape[0]
        # predict_a is for class 9 majority
        else:
            correct_count = \
                test_targets[a_i_9].shape[0] + test_targets[b_i_4].shape[0]
        test_accuracy = correct_count / test_data.shape[0]
        print("Test accuracy is: ", test_accuracy)

        # 5. Plot w as a 28×28image.



    # load the corresponding  4,9 class data
    train_data, train_targets, test_data, test_targets = load_data()

    # add columns of ones to x_data
    train_data = add_bias(train_data)
    test_data = add_bias(test_data)

    # init optimizer as directed
    optimizer_1 = GDOptimizer(lr=0.05, beta=0.0)
    optimizer_2 = GDOptimizer(lr=0.05, beta=0.1)

    penalty = 1.0
    batchsize = 100
    iters = 500
    svm_1 = optimize_svm(train_data, train_targets, penalty,
                     optimizer_1, batchsize, iters)
    svm_2 = optimize_svm(train_data, train_targets, penalty,
                         optimizer_2, batchsize, iters)








def part2_1():
    optimizer_1 = GDOptimizer(lr=1.0, beta=0.0)
    w_history_1 = optimize_test_function(optimizer_1, w_init=10.0, steps=200)

    optimizer_2 = GDOptimizer(lr=1.0, beta=0.9)
    w_history_2 = optimize_test_function(optimizer_2, w_init=10.0, steps=200)

    plt.figure(1)

    line1, = plt.plot(w_history_1, 'b--',label='beta = 0')
    line2, = plt.plot(w_history_2, 'r--', label='beta = 0.9')
    first_legend = plt.legend(handles=[line1, line2], loc=1)

    plt.ylabel('w history points')
    plt.xlabel('number of iteration')
    plt.show()



if __name__ == '__main__':
    # this will demo the GDOptimizer function
    # part2_1()

    pass