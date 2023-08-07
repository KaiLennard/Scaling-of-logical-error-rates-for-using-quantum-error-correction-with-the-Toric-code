import time

# import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import hstack, kron, eye, csr_matrix, block_diag
from pymatching import Matching
from multiprocessing import Pool
import h5py
import os
from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def create_gauss_p_matrix(p_, dev, trials, n_qubits):
    trials_ = int(trials)
    n_qubits_ = int(n_qubits)
    p_all = np.zeros((trials_,n_qubits_))
    for i in range(0,trials_):
        gauss = get_truncated_normal(mean=p_, sd=dev, low=0.00001, upp=0.4)
        p_all[i,:] = gauss.rvs(n_qubits_)
    return p_all

def square_elements_list(list1):
    result = []
    for i in range(0, len(list1)):
        j = i - 1
        # print("element of list ",j, len(list1))
        result.append(list1[j] * list1[j])
    return result


def repetition_code(n):
    """
    Parity check matrix of a repetition code with length n.
    """
    # print('here')
    row_ind, col_ind = zip(*((i, j)
                             for i in range(n)
                             for j in (i, (i + 1) % n)))
    # print(row_ind,col_ind)
    # print("new")

    data = np.ones(2 * n, dtype=np.uint8)
    # print(data)
    # return csr_matrix((data, (row_ind, col_ind)), shape = (n,n)).toarray()
    return csr_matrix((data, (row_ind, col_ind)), shape=(n, n)).toarray()
    # without the shape part


def toric_code_x_stabilisers(L):
    """
    Sparse check matrix for the X stabilisers of a toric code with
    lattice size L, constructed as the hypergraph product of
    two repetition codes.
    """
    Hr = repetition_code(L)
    # print(Hr.shape())
    # print(Hr)
    # print('toric')

    Hx = hstack(
        [kron(Hr, eye(Hr.shape[1])), kron(eye(Hr.shape[0]), Hr.T)],
        dtype=np.uint8
    )
    Hx.data = Hx.data % 2
    Hx.eliminate_zeros()
    return csr_matrix(Hx)


def toric_code_z_stabilisers(L):
    """
    Sparse check matrix for the X stabilisers of a toric code with
    lattice size L, constructed as the hypergraph product of
    two repetition codes.
    """
    Hr = repetition_code(L)
    Hz = hstack(
        [kron(eye(Hr.shape[1]), Hr), kron(Hr.T, eye(Hr.shape[0]))],
        dtype=np.uint8
    )

    Hz.data = Hz.data % 2
    Hz.eliminate_zeros()
    return csr_matrix(Hz)


def toric_code_x_logicals(L):
    H1 = csr_matrix(([1], ([0], [0])), shape=(1, L), dtype=np.uint8)
    # print('H1:')
    # print(H1)
    H0 = csr_matrix(np.ones((1, L), dtype=np.uint8))
    # print('H0:')
    # print(H0)
    x_logicals = block_diag([kron(H1, H0), kron(H0, H1)])
    # print('x_log.data:')
    # print(x_logicals.data)
    x_logicals.data = x_logicals.data % 2
    x_logicals.eliminate_zeros()
    return csr_matrix(x_logicals)


def toric_code_z_logicals(L):
    num_qub = 2 * L * L
    z_logicals = np.zeros((2, num_qub))
    z_logicals[0, L * L: L * L + L] = 1
    z_logicals[1, 0: L * L: L] = 1
    return csr_matrix(z_logicals)


def get_error_prop(number_general_errors, p, number_qubits):
    for i in range(0, number_qubits):
        noise_ = int(np.random.binomial(1, p))

    return noise_


# estimate number of logical errors for num_trials
def num_decoding_failures(Hx_new, Hz_new, x_logicals_new, z_logicals_new, dev_, p_, number_qubits_new,
                          number_general_errors_new, j_start, j_end, p_act_m, p_cal_m):
    # do not forget number of error types here
    noiseAll = np.zeros((number_general_errors_new, number_qubits_new))  # creating noise/error matrix
    num_errorsAll = 0  # set error counter to 0
    p_l2_all = []
    # alphaX = p_*dev
    # alphaid = dev *(1- p_)
    # print(np.log((1 - p_) / p_))
    # print(np.shape(np.full((number_qubits_new), np.log((1 - p_) / p_))))

    # weights_new = np.full((number_qubits_new), np.log((1-p_cal_)/p_cal_)) #compute weights vector with p_cal_
    # print(np.shape(weights_new))

    # matchingx = Matching(graph=Hz_new, weights=weights_new) #creating matching objects with parity check matrix and weights of qubits

    np.random.seed((os.getpid() * int(
        time.time())) % 123456789)  # random seed so that each process computes independent random errors

    for j in range(j_start, j_end):
        # X errors
        p_cal_ = p_cal_m[j,:]
        p_act_ = p_act_m[j,:]

        weights_new = np.log((1 - p_cal_) / p_cal_)  # compute weights vector with p_cal_
        #print("lets look here ",weights_new)
        matchingx = Matching(graph=Hz_new,
                             weights=weights_new)  # creating matching objects with parity check matrix and weights of qubits

        noiseAll[1, :] = np.random.binomial(1, p_act_, Hz_new.shape[1])
        # print(p_act_)# fill in error X vector with probability p_act_
        # print(noiseAll[1,:])
        # dectect X error using Z stabilizer

        syndromex = (Hz_new @ noiseAll[1, :]) % 2  # compute syndrome by matrix multi. of Hz and noise/error vector

        correctionx = matchingx.decode(z=syndromex, num_neighbours=None)  # decode computes corrections

        errorx = (noiseAll[1, :] + correctionx) % 2  # apply corrections
        if (np.any(errorx @ z_logicals_new.T % 2)):  # check if all corrections were successful
            num_errorsAll += 1  # if unsuccessful add 1 to error counter
    return num_errorsAll  # return error counter


remain_errors = []  # error counting array for processes


def log_result(result):
    # print("add")
    remain_errors.append(result)


if __name__ == '__main__':
    """26.06.2023"""
    start = time.time()  # measure time

    num_trials = 1e5  # number of repetitions
    print("num_trials = " + str(num_trials))
    deviation = [0.001, 0.01,0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]  # deviation of gauss for probability
    #Ls = range(5, 24, 2)  # lattice size
    Ls = [5, 15, 25, 35, 45, 55]
    # ps = [0.01,0.02,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.1,0.11] #probabilities around the threshold where all qubits have the same error rate within a lattice of size L
    # ps = np.arange(0.05,0.12,0.005)
    # ps = np.around(ps,5)
    Ps = [0.09]
    # print(ps[2])
    #print(np.shape(p))
    #print(p)
    np.random.seed(2)

    number_errors = 2  # kinds of errors (if two only id and X errors appear)
    number_processes = 50  # multiprocessing.cpu_count()
    print("number of processes/cores = " + str(number_processes))
    for p in Ps:
        print(p, "==== new error rate p ####")
        #print("+++++++++++++++++++++++++++++++++++ dev=" + str(dev))
        start_dev = time.time()
        log_errors_all_L_sum = []
        log_errors_all_L_norm = []
        log_errors_all_L_sqrt = []
        # array to store all error rates for one deviation
        p_mean_all_L = []
        p_var_all_L = []
        L2_norm__all_L = []

        print(p)
        for L in Ls:
            start_L = time.time()
            print("NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW  Simulating L={}...".format(L))
            Hx = toric_code_x_stabilisers(L)  # computing parity check matrices for a lattice of size L
            Hz = toric_code_z_stabilisers(L)
            logX = toric_code_x_logicals(L)  # computing logical operators for a lattice of size L
            logZ = toric_code_z_logicals(L)
            log_errors_sum = []
            log_errors_norm = []
            log_errors_sqrt = []
            p_mean_L = []
            p_var_L = []
            L2_norm_L = []
            number_of_Qubits = 2 * L * L

            for dev in deviation:
                p_matrix = np.zeros((number_errors, number_of_Qubits))
                p_matrix[1, :] = p  # creating array of p to using in gauss function
                remain_errors = []
                #print("portion = ", por)
                pool = Pool()  # create pool for parallel processes
                p_cal = create_gauss_p_matrix(p, dev, num_trials, number_of_Qubits)
                p_act = create_gauss_p_matrix(p, dev, num_trials, number_of_Qubits)

                print("p=", p, " L = ", L, " dev = ", dev, "of ", deviation[len(deviation) - 1])
                for proc in range(number_processes):
                    trial_start = int(
                        proc * num_trials / number_processes)  # the num_trials are computed by parallel processes, where each process computes a equal number of trials
                    trial_end = int((proc + 1) * num_trials / number_processes)
                    # print("here before aply async")

                    pool.apply_async(num_decoding_failures, args=(
                    Hx, Hz, logX, logZ, dev, p, number_of_Qubits, number_errors, trial_start, trial_end, p_act, p_cal),
                                     callback=log_result)  # the parallel processing command, callback calls the function log_results so each process saves its result independently but in the same array
                    # print(remain_errors)

            #print(sum(num_decoding_failures(Hx, Hz, logX, logZ, por, p, number_of_Qubits, number_errors, trial_start, trial_end, dev)))
                pool.close()
                pool.join()  # close all parallel processes
                print(str(sum(remain_errors)) + "  = remain errors         ")

                #print("squared elements = ", square_elements_list(remain_errors))
                #print("sum = ", sum(square_elements_list(remain_errors)))
                #print("normal = ", sum(remain_errors) / num_trials)
                #print("sqrt (sum) = ", np.sqrt(sum(square_elements_list(remain_errors))))

                log_errors_sqrt.append(np.sqrt(sum(square_elements_list(
                    remain_errors))) / num_trials)  # logical error rate is saved for each p is saved
                print("log errors = ", log_errors_sqrt[len(log_errors_sqrt) - 1], "log errors done")
                log_errors_norm.append(sum(remain_errors) / num_trials)
                log_errors_sum.append(sum(remain_errors))
                temp = np.zeros(len(p_act[:,0]))
                for i in range(0,len(p_act[:,0])):
                    temp[i] = np.average(square_elements_list(p_act[i,:]))

                L2_norm_L.append(np.sqrt(np.average(temp))/(num_trials))

            log_errors_all_L_sqrt.append(np.array(log_errors_sqrt))  # logical error rate for each L is saved
            log_errors_all_L_sum.append(np.array(log_errors_sum))
            log_errors_all_L_norm.append(np.array(log_errors_norm))
            L2_norm__all_L.append(L2_norm_L)
            # print(log_errors_all_L)
            end_L = time.time()
            print("time for (L = " + str(L) + ") = " + str((end_L - start_L) / 60))

        end_dev = time.time()
        print("time (min):")
        print((end_dev - start_dev) / 60, "min per dev")  # print out time
        print("all errors = " + str(log_errors_all_L_sqrt))

        f1 = h5py.File("data_23_07_11_GAUSS_BD_p_"+str(p)+ ".hdf5", "w")  # save data in hdf5 file
        dset1 = f1.create_dataset("dataset", np.shape(log_errors_all_L_sqrt), dtype='d', data=log_errors_all_L_sqrt)
        dset1.attrs['trials'] = num_trials
        dset1.attrs['properties'] = ['l2 norm/ num_trials angegeben']
        dset1.attrs['ps'] = p
        dset1.attrs['Ls'] = Ls
        dset1.attrs['dev'] = dev
        dset1.attrs['L2_norm'] = L2_norm__all_L
        dset1.attrs['num_trials'] = num_trials
        dset1.attrs['log_errors_sqrt'] = log_errors_all_L_sqrt
        dset1.attrs['log_errors_sum'] = log_errors_all_L_sum
        dset1.attrs['log_errors_norm'] = log_errors_all_L_norm
    end = time.time()
    print((end - start) / 60, "minutes over all")  # print out time
