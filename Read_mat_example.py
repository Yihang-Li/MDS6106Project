import scipy.io
from scipy.sparse import csr_matrix
mat = scipy.io.loadmat('datasets/datasets/a9a/a9a_test.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
# mat: {'A': <16281x122 sparse matrix of type '<class 'numpy.float64'>'
 	#with 225731 stored elements in Compressed Sparse Column format>}
print(mat['A'])
S = mat['A']
# convert back to 2-D representation of the matrix
D = S.todense()
print("Dense matrix: \n", D)


def read_mat_data(direct):
    import scipy.io
    from scipy.sparse import csc_matrix
    mat = scipy.io.loadmat(direct)
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    # mat: {'A': <16281x122 sparse matrix of type '<class 'numpy.float64'>'
    # with 225731 stored elements in Compressed Sparse Column format>}
    # print(mat['A'])
    S = mat['A']
    # convert back to 2-D representation of the matrix
    select_train = np.array([i for i in range(S.shape[1] - 1)])
    select_test = np.array([S.shape[1] - 1])
    train = S.tocsc()[:, select_train]
    test = S.tocsc()[:, select_test]
    # D = S.todense()
    # print(train.shape, test.shape,S.shape)
    # print("Dense matrix: \n", D)
    # print(type(S),type(train),type(test),S.shape,train.shape,test.shape)
    return train, test


a, b = read_mat_data('datasets/datasets/a9a/a9a_test.mat')