import time
import utils
import os
import numpy as np

# Draft Version: randomly sample the source domain
# TO DO: try sampling on different target domains
class replay_buffer:
    # data_dir: the fold where the replay buffer will draw images from
    # portion: the portion of files in data_dir will form replay buffer
    def __init__(self, data_dir, portion):
        self.capacity = int(portion * len(os.listdir(data_dir)))
        self.images, self.labels = utils.sample_batch(data_dir, batch_size = self.capacity)

        
# Random sample from the replay buffer
# sample_size: the number of images & labels selected from buffer
def random_sample_replay(replay_buffer, sample_size):
    indicies = np.random.choice(range(replay_buffer.capacity), sample_size, replace=False)
    
    X = []
    Y = []
    for i in indicies:
        X.append(replay_buffer.images[i])
        Y.append(replay_buffer.labels[i])
    
    return np.asarray(X), np.asarray(Y)


#Read an input file, which includes target data information: (index, entropy, label)
def read_entropy_file(filename):
    index = []
    entropy = []
    label = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            tokens = line.split()
            index.append(int(tokens[0]))
            entropy.append(float(tokens[1]))
            label.append(int(tokens[2]))
            entropy = np.array(entropy)
    return index, entropy, label

#Random selection
def get_proxy_data_random(entropy, sampling_portion, logging=None):
    num_proxy_data = int(np.floor(sampling_portion * len(entropy)))
    indices = np.random.choice(range(len(entropy)), size=num_proxy_data, replace=False)

    np.random.shuffle(indices)
    return indices

