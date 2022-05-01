'''
    The file contains various utils that are not specific enough to fall into the other groups of functions. 
'''

import tensorflow as tf

import numpy as np
import os

import io_utils as io

def learn_gaussians(data_dir, z_model, sm_model, batch_size, label_ids, rho=.97, initial_means=None):
    '''
        Code to learn gaussians for domain adaptation.

        data_dir - directory of images and labels
        z_model - model predicting latent features for the data
        sm_model - softmax activation layer
        batch_size - size of batch to be used
        label_ids - dictionary of label_name:label_id
        rho - threshold parameter for considering a latent feature for the GMM construction
        initial_means - to corretly compute the variance of the data, we first need to compute its means

    '''
    num_classes = len(label_ids)
    
    means = initial_means
    if initial_means is None:
        means = np.zeros((num_classes, num_classes))
        
    covs = np.zeros((num_classes, num_classes, num_classes))
    cnt = np.zeros(num_classes)
    
    start = 0
    
    # total number of images
    N = len(os.listdir(data_dir)) // 2
    
    while start < N:
        # Read in current batch
        data = []
        y_true = []
        for i in range(start, start+batch_size):
            try:
                data.append(io.read_image(data_dir + str(i) + "_image.npy"))
                y_true.append(io.read_label(data_dir + str(i) + "_label.npy"))
            except:
                print("Image", data_dir + str(i) + "_image.npy", "could not be processed")
                
        data = np.asarray(data)
        y_true = np.asarray(y_true)
        
        # Predict latent features
        zs = z_model.predict(data).reshape(-1, num_classes)
        
        # Get softmax outputs
        y_hat = sm_model.predict(data).reshape(-1, num_classes)
        vmax = np.max(y_hat, axis=-1)
        y_hat = np.argmax(y_hat, axis=-1)
        
        y_t = y_true.ravel()

        # Keep a few exemplars per class
        for label in label_ids:
            if label == 'ignore':
                continue

            c = label_ids[label]
            ind = (y_t == c) & (y_hat == c) & (vmax > rho)
            
            if np.sum(ind) > 0:
                # We have at least one sample
                
                # Reshape to make sure dimensions stay the same
                curr_data = zs[ind].reshape(-1, num_classes)
                
                if initial_means is None:
                    # Only update means and counts
                    
                    means[c] += np.sum(curr_data, axis=0)
                    cnt[c] += np.sum(ind)
                else:
                    # ! here means are scaled to their final values
                    sigma = np.dot(np.transpose(curr_data - means[c]), curr_data - means[c])
                    assert sigma.shape == (num_classes, num_classes)
                    covs[c] += sigma
                    
                    cnt[c] += np.sum(ind)
                    
        start += batch_size
        
    # Normalize results
    for i in range(num_classes):
        if i == 0:
            continue
        
        if initial_means is None:
            means[i] /= cnt[i]
        covs[i] /= (cnt[i] - 1)
    
    return means, covs, cnt

def sample_from_gaussians(means, covs, n_samples):
    # Return samples from the num_classes gaussians trained on the source dataset
    # n_samples is an array of the same size as gmms, representing the number of samples to
    # be returned from each gaussian
    
    # class 0 is the class to be ignored
    assert n_samples[0] == 0
    
    n = len(n_samples)
    
    res_x = []
    res_y = []
    
    for i in range(n):
        if n_samples[i] > 0:
            curr_x = np.random.multivariate_normal(means[i], covs[i], n_samples[i])
            curr_y = np.repeat(i, n_samples[i])
                        
            res_x.append(curr_x)
            res_y.append(curr_y.reshape(-1,1))

    res_x = np.vstack(res_x)
    res_y = np.vstack(res_y).ravel()
    
    perm = np.random.permutation(res_x.shape[0])
    
    return res_x[perm,:], res_y[perm]


def compute_miou(data_dir, model, label_ids, ids_to_ignore = [0]):
    # Returns the mean IoU for every class, and averages over all classes
    # label_ids is a dict of label_name -> label_idx
    
    N = len(os.listdir(data_dir)) // 2
    
    intersection = dict()
    union = dict()
    for label in label_ids:
        intersection[label] = union[label] = 0
        
    for i in range(N):
        X = np.array([io.read_image(data_dir + str(i) + "_image.npy")])

        y_true = io.read_label(data_dir + str(i) + "_label.npy").ravel()
        y_hat = model.predict(X).ravel()
        
        for label in label_ids:
            if label_ids[label] in ids_to_ignore:
                continue
                
            curr_id = label_ids[label]
            
            idx_gt = y_true == curr_id
            idx_hat = y_hat == curr_id
            
            intersection[label] += np.sum(idx_gt & idx_hat)
            union[label] += np.sum(idx_gt | idx_hat)
        
    mIoU = []
    res = dict()
    for label in label_ids:
        if label_ids[label] in ids_to_ignore:
            continue
            
        if union[label] != 0:
            res[label] = intersection[label] / union[label]
        else:
            res[label] = np.float64(0)

        mIoU.append(res[label])

    return res, np.mean(mIoU)

def get_predictions(dataset, model, batch_size):
    # Given a dataset and a keras model, returns the predictions of the model on said dataset
    # If the dataset is too large to pass at once to the model, it is split in batches of size batch_size

    res = []
    
    start = 0
    end = batch_size
    
    while start < dataset.shape[0]:
        res.append(model.predict(dataset[start:end]))
        
        start += batch_size
        end += batch_size
    
    return np.vstack(res)

def sample_batch(data_dir, batch_size = 4, seed = None):
    '''
        Samples a batch of data
    '''

    if seed is not None:
        np.random.seed(seed)
    
    N = len(os.listdir(data_dir)) // 2
    
    ind = np.random.choice(range(N), batch_size, replace=False)
    
    X = []
    Y = []
    
    for i in ind:
        X.append(io.read_image(data_dir + str(i) + "_image.npy"))
        Y.append(io.read_label(data_dir + str(i) + "_label.npy"))

    # Return the image/label pairs    
    return np.asarray(X), np.asarray(Y)
