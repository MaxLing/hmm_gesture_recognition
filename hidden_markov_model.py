'''
Note: Baum-Welch Algorithm, calculation is in log space to prevent underflow
'''
import numpy as np
import pickle
from scipy.special import logsumexp

def hmm_train(trans, emiss, init, max_iter, tol, observations, gestures):
    hmm_models = {}
    M, N = emiss.shape
    for gesture in gestures:
        A = np.log(trans)
        B = np.log(emiss)
        PI = np.log(init)
        obs_seqs = [observations[key] for key in observations.keys() if gesture in key] # multiple observation sequences
        K = len(obs_seqs)

        last_log_likelihood = None
        for i in range(max_iter):
            first_gamma = []
            all_gamma = []
            all_xi = []
            all_obs_st = []
            log_likelihood = np.zeros(K)

            for idx, obs in enumerate(obs_seqs):
                ## E-step
                # computer statistics of hidden states
                alpha, beta, P = forward_backward(obs, A, B, PI)
                log_likelihood[idx] = P
                gamma = smoothing(alpha, beta) # N*T
                xi = pair_states(alpha, beta, A, B, obs) # N*N*(T-1)

                first_gamma.append(gamma[:,0])
                all_gamma.append(logsumexp(gamma, axis=1))
                all_xi.append(logsumexp(xi, axis=2))

                obs_st = np.zeros((M,N))
                for m in range(M):
                    try:
                        obs_st[m] = logsumexp(gamma[:, obs == m], axis=1)
                    except ValueError: # no observation m in sequence
                        obs_st[m] = np.full(N, -np.inf)
                all_obs_st.append(obs_st)

            all_log_likelihood = np.sum(log_likelihood)
            print(gesture, '\tIteration: ', i+1, '\tlog-likelihood: ', all_log_likelihood, '\tall: ', log_likelihood)

            first_gamma_sum = logsumexp(np.array(first_gamma), axis=0)
            gamma_sum = logsumexp(np.array(all_gamma), axis=0).reshape((1,-1))
            xi_sum = logsumexp(np.array(all_xi), axis=0)
            obs_st_sum = logsumexp(np.array(all_obs_st), axis=0)

            ## M-step
            # reestimate parameters that best fit inferred distribution
            A = xi_sum - gamma_sum
            B = obs_st_sum - gamma_sum
            PI = first_gamma_sum
            # normalize
            A -= logsumexp(A, axis=0)
            B -= logsumexp(B, axis=0)
            PI -= logsumexp(PI)

            # check covergence
            if last_log_likelihood is not None and abs(all_log_likelihood - last_log_likelihood) < tol:
                break
            last_log_likelihood = np.sum(log_likelihood)

        # save hmm parameters under right gesture
        hmm_models[gesture] = {'prior': PI, 'transition': A, 'emission': B}

    # save the whole model
    pickle.dump(hmm_models, open("hmm_models.p", "wb"))

def forward_backward(obs, A, B, PI):
    M, N = B.shape
    T = obs.shape[0]

    # Forward procedure
    # init
    alpha = np.zeros((T,N))
    alpha[0] = PI + B[obs[0]]
    # induction
    for t in range(T-1):
        alpha[t+1] = logsumexp(alpha[t].reshape((1,-1)) + A, axis=1) + B[obs[t+1]]
    # termination, compute likelihood
    P = logsumexp(alpha[T-1])

    # Backward procedure
    # init
    beta = np.zeros((T,N))
    beta[T-1] = np.log(1)
    # induction
    for t in range(T-2, -1, -1):
        beta[t] = logsumexp(beta[t+1].reshape((-1,1)) + A + B[obs[t + 1]].reshape((-1,1)), axis=0)
    # termination, for debug purpose
    # Q = logsumexp(beta[0]+PI)
    # Note: P != Q because Q doesn't include the first observation! Thus P < Q and the difference should be small
    # print('forward log likelihood: ', P, '\tbackward log likelihood: ', Q)

    return alpha, beta, P

def smoothing(alpha, beta):
    # T, N = alpha.shape
    # gamma = np.zeros((T,N))
    #
    # for t in range(T):
    #     gamma[t] = alpha[t] + beta[t]
    gamma = (alpha + beta).T
    gamma -= logsumexp(gamma, axis=0).reshape((1,-1)) # normalization

    return gamma

def pair_states(alpha, beta, A, B, obs):
    T, N = alpha.shape
    xi = np.zeros((N, N, T-1))

    for t in range(T-1):
        xi[:, :, t] = alpha[t].reshape((1,-1)) + A + B[obs[t+1]].reshape((-1,1)) + beta[t+1].reshape((-1,1))
    xi -= logsumexp(xi, axis=2).reshape((N,N,1)) # normalization
    xi[np.isnan(xi)] = -np.inf # replace nan with -inf, triu
    return xi

