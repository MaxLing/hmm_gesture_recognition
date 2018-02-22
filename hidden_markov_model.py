import numpy as np
from numpy import log
from scipy.special import logsumexp

def hmm_train(trans, emiss, init, max_iter, tol, data):
    M, N = emiss.shape
    for gesture in data['gestures']:
        A = np.copy(trans)
        B = np.copy(emiss)
        PI = np.copy(init)
        obs_seqs = [data[key] for key in data.keys() if gesture in key] # multiple observation sequences

        for i in range(max_iter):
            gamma = np.zeros(N)
            xi = np.zeros((N, N))
            log_likelihood = np.zeros(len(obs_seqs))

            for idx, obs in enumerate(obs_seqs):
                ## E-step
                # params in log space: alpha, beta, P, gamma, xi
                alpha, beta, P = forward_backward(obs, A, B, PI)
                log_likelihood[idx] = P
                gamma = smoothing(alpha, beta)
                xi = pair_states(alpha, beta, A, B, obs)

            mean_log_likelihood = np.mean(log_likelihood)
            print(gesture, '\tIteration: ', i+1, '\tAverage log-likelihood: ', mean_log_likelihood)

            ## M-step




        print(0)

def forward_backward(obs, A, B, PI):
    M, N = B.shape
    T = obs.shape[0]

    # Forward procedure
    # init
    alpha = np.zeros((T,N))
    alpha[0] = log(PI)+log(B[obs[0]])
    # induction
    for t in range(T-1):
        alpha[t+1] = logsumexp(alpha[t]+log(A), axis=1) + log(B[obs[t+1]])
    # termination, compute likelihood
    P = logsumexp(alpha[T-1])

    # Backward procedure
    # init
    beta = np.zeros((T,N))
    beta[T-1] = log(1)
    # induction
    for t in range(T-2, -1, -1):
        beta[t] = logsumexp(beta[t+1]+log(A.T)+log(B[obs[t+1]]), axis=1)
    # termination, for debug purpose
    Q = logsumexp(beta[0]+log(PI))
    # TODO: figure out P == Q
    # print('forward log likelihood: ', P, '\tbackward log likelihood: ', Q)
    return alpha, beta, P

def smoothing(alpha, beta):
    T, N = alpha.shape
    gamma = np.zeros((T,N))

    for t in range(T):
        gamma[t] = alpha[t] + beta[t]
    gamma -= logsumexp(gamma, axis=1).reshape((-1,1)) # normalization

    return gamma

def pair_states(alpha, beta, A, B, obs):
    T, N = alpha.shape
    xi = np.zeros((N, N, T-1))

    for t in range(T-1):
        xi[:, :, t] = alpha[t]+log(A)+log(B[obs[t+1]])+beta[t+1]
    xi -= logsumexp(xi, axis=2).reshape((N,N,1)) # normalization

    return xi

# def log_baum_welch(observations):
