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