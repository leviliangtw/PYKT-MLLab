from numpy import array, cov, mean
from numpy.linalg import eig

A = array([[1, 2, 3],
           [3, 4, 5],
           [5, 6, 7],
           [7, 8, 9]])
print("A: \n", A)
M = mean(A.T, axis=1)  # x-axis, row
print("M: ", M)
M2 = mean(A.T)
print("M2: ", M2)
M3 = mean(A, axis=1)
print("M3: ", M3)
C = A - M
print("C: \n", C)
V = cov(C.T)
print("V: \n", V)
values, vectors = eig(V)
print("eigen values: ", values)
print("eigen vectors: \n", vectors)
p = vectors.T.dot(C.T)
print("project array:\n", p.T)
