import os, pickle
import numpy as np
from sklearn.cluster import KMeans
from hidden_markov_model import *

def main():
    ''' modify this part accordingly '''
    train_path = 'train_data'
    test_path = 'test_data'
    N = 10 # num of hidden states
    M = 30 # num of observation classes
    # initialization
    PI = np.ones(N)/N
    # transition
    A = np.random.uniform(low=0.1, high=1, size=(N, N))
    A = np.tril(A, k=0) # take lower triangle, allow model to stay or go right
    A /= np.sum(A, axis=0)
    # emission
    B = np.random.uniform(low=0.1, high=1, size=(M, N))
    B /= np.sum(B, axis=0)
    # hmm training params
    max_iter = 10
    tol = 0.01

    # feature extraction and clustering
    raw = load_data(train_path)
    # TODO: extract more features?
    cluster = cluster_data(raw, M, test = False)

    # train hmm
    hmm_train(A, B, PI, max_iter, tol, cluster)

    # test hmm

    print(0)

def load_data(path):
    data = {}
    gestures = []
    for file in os.listdir(path):
        filename = os.path.splitext(file)[0]
        gesture = filename.split('_')[0]

        # save gesture classes
        if gesture not in gestures:
            gestures.append(gesture)

        # load sensor data
        raw_data = np.loadtxt(os.path.join(path, file))

        data[filename] = raw_data[:,1:] # ts not used
    data['gestures'] = gestures
    return data

def cluster_data(data, K, test):
    if not test: # K-means on train data
        all_data = np.array([])
        for idx, key in enumerate(data):
            if key is 'gestures':
                continue
            if idx == 0:
                all_data = data[key]
            all_data = np.vstack((all_data,data[key]))
        kmeans = KMeans(n_clusters=K).fit(all_data)
        pickle.dump(kmeans, open("K_means.p", "wb"))
    else:  # load K-means from train data
        kmeans = pickle.load(open("K_means.p", "rb"))
    # cluster observation classes
    for key, value in data.items():
        if key is 'gestures':
            continue
        data[key] = kmeans.predict(value)

    return data



if __name__ == '__main__':
    main()
