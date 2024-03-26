import numpy as np
from scipy.stats import qmc
import pickle

def lidar_data(xy, label, complete=0, n=1):
    n_test = 1000
    if n == 1:
        # Homogenous partial
        if complete==0:
            # Random 50% for training
            X_train = np.float_(xy[::1, :2])
            Y_train = np.float_(label[::1, 0][:, np.newaxis]).ravel()  # * 2 - 1
            
            # 20% for testing
            X_test = np.float_(xy[::7, :2])
            Y_test = np.float_(label[::7, 0][:, np.newaxis]).ravel() 
                
        # Homogenous complete
        elif complete==1:
            raise ("NotImplementedError")
            # 90% for training
            X_train = np.float_(g[np.mod(np.arange(len(g)), 10) != 0, 1:3])
            Y_train = np.float_(g[np.mod(np.arange(len(g)), 10) != 0, 3][:, np.newaxis]).ravel()  # * 2 - 1
            
            # 10% for testing
            X_test = np.float_(g[::10, 1:3])
            Y_test = np.float_(g[::10, 3][:, np.newaxis]).ravel() 
            
        else:
            raise ValueError('Use 0 or 1.')
    else:
        raise ("NotImplementedError")
        # Distributed homogenous
        if complete == 0:
            assert n <= 50
            # Lower bound of the number len(g)
            size_ = int(len(g)/n)
            data_len = n*size_
            # Consists of numbers from 0 to n-1
            remainder_list = np.mod(np.arange(len(g)), n)
            remainder_list[data_len:] = n
            
            X_train = np.zeros((n, size_, 2))
            Y_train = np.zeros((n, size_))
            for idx in range(n):
                X_train[idx, :, :] = np.float_(g[remainder_list == idx, 1:3])
                Y_train[idx, :] = np.float_(g[remainder_list == idx, 3][:, np.newaxis]).ravel()  # * 2 - 1
                
            # 10% for testing
            X_test = np.float_(g[::n, 1:3])
            Y_test = np.float_(g[::n, 3][:, np.newaxis]).ravel() 
        
        else:
            assert n <= 10
            # Lower bound of the number len(g)
            size_ = int(len(g)/n)
            data_len = n*size_
            
            X_train = np.zeros((n, size_, 2))
            Y_train = np.zeros((n, size_))
            for idx in range(n):
                X_train[idx, :, :] = np.float_(g[idx*size_:(idx+1)*size_, 1:3])
                Y_train[idx, :] = np.float_(g[idx*size_:(idx+1)*size_, 3][:, np.newaxis]).ravel()  # * 2 - 1
                
            # 10% for testing
            X_test = np.float_(g[::n, 1:3])
            Y_test = np.float_(g[::n, 3][:, np.newaxis]).ravel() 
        
    n_test = min(n_test, X_test.shape[0])
    test_idx = np.random.randint(0, X_test.shape[0], n_test)
    X_ver = X_test[test_idx, :]
    Y_ver = Y_test[test_idx]
        
    # print(len(label), len(Y_train), len(Y_test))
    return X_train, Y_train, X_test, Y_test, X_ver, Y_ver
    
    
def points(xy, label, nf, type_ = 'random'):
    x_min, x_max = min(xy[:,0]), max(xy[:,0])
    y_min, y_max = min(xy[:,1]), max(xy[:,1])
    
    # First value is 1 for bias term
    lscale = np.ones(nf+1)
    if type_ == 'random':
        sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
        fpoints = sampler.random(n=nf)
        fpoints[:,0] = x_min+(x_max-x_min)*fpoints[:,0]
        fpoints[:,1] = y_min+(y_max-y_min)*fpoints[:,1]
        
        return fpoints, lscale
    else:
        fpoints = np.zeros((nf, xy.shape[1]))
        # Randomized assignments
        n1 = int(0.75*nf)
        f_idx = np.random.randint(0, len(xy), n1)
        fpoints[:n1,:] = xy[f_idx, 0:2]
        for r_idx, idx in enumerate(f_idx):
            if label[idx, 0] != 1:
                lscale[r_idx+1] = 0.3
            else:
                lscale[r_idx+1] = 0.1
        # Adding extra wall features
        n_indices = np.where(np.array(label)>0.5)[0]
        f_idx = np.random.randint(0, len(n_indices), int(0.25*nf))
        # print (n_indices, n_indices[f_idx])
        fpoints[n1:,:] = xy[n_indices[f_idx], 0:2]
        for r_idx, idx in enumerate(n_indices[f_idx]):
            if label[idx, 0] != 1:
                lscale[n1+r_idx+1] = 0.05
            else:
                lscale[r_idx+1] = 0.05
        return fpoints, lscale

def mix_points(xy, labels, nf):
    alfa = 0.6
    fpoints1, lscale1 = points(xy, labels, nf=int(alfa*nf), type_ = 'random')
    fpoints2, lscale2 = points(xy, labels, nf=nf-int(alfa*nf), type_ = 'selected')
    fpoints = np.vstack((fpoints1, fpoints2))
    lscale = np.hstack((lscale1, lscale2[1:]))
    return fpoints, lscale


def gen(path: str):
    lines = []
    with open(path,'r') as f:
        lines = f.readlines()
    data_len = len(lines)
    coords = np.zeros((data_len, 2))
    labels = np.zeros((data_len, 1))
    for k in range(data_len):
        val = [float(i) for i in str(lines[k]).split(',')]
        coords[k,:] = val[:2]
        labels[k,0] = val[2]
    xy_coords_skip = coords[::5,:2]
    labels_skip = labels[::5,:2]

    X_train, Y_train, X_test, Y_test, X_ver, Y_ver = lidar_data(xy_coords_skip, labels_skip, complete=0, n=1)

    fpoints, lscale = mix_points(xy_coords_skip, labels_skip, nf=280)
    with open(path.split('scannedPoints.txt')[0]+'fpl.pcl', 'wb') as handle:
        pickle.dump([fpoints, lscale, X_ver, Y_ver, X_test], handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    gen('scannedPoints.txt')
    
if __name__== '__main__':
    main()