import os, pickle
import numpy as np
from sklearn.cluster import KMeans
from hidden_markov_model import *

def main():
    ''' modify this part accordingly '''
    TRAIN = False
    PREDICT = True
    train_path = 'train_data'
    test_path = 'test_data'

    if TRAIN:
        N = 10 # num of hidden states
        M = 15 # num of observation classes
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
        max_iter = 50
        tol = 0.1

        # feature extraction and clustering
        raw = load_data(train_path)
        # TODO: extract more features?
        # TODO: tune M at the same time, prevent too much -inf in B
        k_means(raw, M)
        cluster = cluster_data(raw)

        # train hmm
        hmm_train(A, B, PI, max_iter, tol, cluster)

    if PREDICT:
        raw = load_data(test_path)
        cluster = cluster_data(raw)
        hmm_predict(cluster)

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

def k_means(data, K):
    # K-means on train data
    all_data = np.array([])
    for idx, key in enumerate(data):
        if key is 'gestures':
            continue
        if idx == 0:
            all_data = data[key]
        all_data = np.vstack((all_data,data[key]))
    kmeans = KMeans(n_clusters=K).fit(all_data)
    pickle.dump(kmeans, open("k_means.p", "wb"))
    return

def cluster_data(data):
    kmeans = pickle.load(open("k_means.p", "rb"))
    # cluster observation classes
    for key, value in data.items():
        if key is 'gestures':
            continue
        data[key] = kmeans.predict(value)
    return data

def hmm_predict(data):
    hmm_models = pickle.load(open("hmm_models.p", "rb"))
    gestures = [key for key in hmm_models.keys()]
    filenames = [key for key in data.keys() if key is not 'gestures']
    instances = len(filenames)
    log_likelihood = np.zeros((instances,len(gestures)))
    for j, gesture in enumerate(gestures):
        # extract params
        prior = hmm_models[gesture]['prior']
        transition = hmm_models[gesture]['transition']
        emission = hmm_models[gesture]['emission']

        for i, filename in enumerate(filenames):
            obs = data[filename]
            _, _, P = forward_backward(obs, transition, emission, prior)
            log_likelihood[i,j] = P

    prediction = [gestures[idx] for idx in np.argmax(log_likelihood, axis=1)]
    print('Instances: ', filenames, '\nPrediction: ', prediction)

    # test accuracy
    correct = [prediction[idx] in filenames[idx] for idx in range(instances)]
    accuracy = np.sum(correct)/instances
    print('Test accuracy: ', accuracy)


if __name__ == '__main__':
    main()
