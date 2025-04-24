

import numpy as np

##### understanding chat gpt code for U v U'
import numpy as np
from scipy.linalg import eigh
def find_closest_vector(Z):
    # Ensure Z is symmetric
    Z = (Z + Z.T) / 2
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = eigh(Z)
    # Find the index of the maximum eigenvalue
    max_eigenvalue_index = np.argmax(eigenvalues)
    # Extract the corresponding eigenvector
    x = eigenvectors[:, max_eigenvalue_index]
    # Normalize the vector to have unit length
    #x /= np.linalg.norm(x)
    return x * eigenvalues[  max_eigenvalue_index  ]
# Example usage
Z = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])
resulting_x = find_closest_vector(Z)
print("Matrix Z:")
print(Z)
print("\nFound vector x:")
print(resulting_x)
print("\nReconstructed x * x^T:")
print(np.outer(resulting_x, resulting_x))









##### understanding support vector decomposition:

z = np.arange(100).reshape((10,10))
u, s, vt = np.linalg.svd(z)
ssqrt = np.tile( np.sqrt( s )  , ( len( s ),1 ))
new_u = ssqrt * u
new_vt = ssqrt.transpose() * vt
print("first:")
print( np.matmul( new_u[:,0:1] , new_vt[0:1,:]  ).round(4) )
print("second:")
print( np.matmul( new_u[:,0:2] , new_vt[0:2,:]  ).round(4))
print("all:")
print( np.matmul( new_u , new_vt  ).round(4))
print("should equal:")
print( np.matmul(u,np.matmul( s* np.identity(len(s))  ,vt)).round(4))
# print( new_u[:,0:1].reshape(-1) - new_vt[0:1,:].reshape(-1)  )

##### understanding eigen decomposition:
# Perform eigenvalue decomposition
z = np.arange(9).reshape((3,3))
eigenvalues, eigenvectors = np.linalg.eig(z)


np.matmul( eigenvectors , np.matmul( np.identity(3)*eigenvalues, eigenvectors.transpose() ))
np.matmul( eigenvectors.transpose() , np.matmul( np.identity(3)*eigenvalues, eigenvectors ))
np.matmul( eigenvectors , np.matmul( np.identity(3)*eigenvalues, np.linalg.inv( eigenvectors ) ))
#well, this isn't as useful, not going to lie








# Choose the two largest eigenvalues and corresponding eigenvectors
largest_indices = np.argsort(eigenvalues)[-2:]
x = eigenvectors[:, largest_indices[0]]
y = eigenvectors[:, largest_indices[1]]

# Reshape x and y to column vectors
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Compute the outer product of x and y
xy_outer = np.outer(x, y)

# Check the approximation
approximation_error = np.linalg.norm(z - xy_outer)

print("Matrix z:")
print(z)
print("\nApproximation using outer product of x and y:")
print(xy_outer)
print("\nApproximation error:", approximation_error)







#### stupid calculus ####

##########################

##########
import sympy as sp


x1,x2,y1,y2,c12=sp.symbols("x1 x2 y1 y2 c12")

obj = ( (x1-x2)**2 + (y1-y2)**2 - c12 )**2

fullDiff = sp.diff(obj,x1)
for var in [x2,y1,y2]:
    fullDiff = fullDiff + sp.diff(obj,var)

print(  fullDiff )
print(  fullDiff.simplify() )

### waste ###
for var in [x1,x2,y1,y2]:
    diff = sp.diff(obj,var)
    print(var.name+":", diff.as_poly() )

######### solving it?


xx' = c


'''
4*x1**3 - 12*x1**2*x2 + 12*x1*x2**2 + 4*x1*y1**2 - 8*x1*y1*y2 + 4*x1*y2**2 - 4*x1*c12 - 4*x2**3 - 4*x2*y1**2 + 8*x2*y1*y2 - 4*x2*y2**2 + 4*x2*c12
x'xx'       x'xx'         x'xx'         xy'y          xy'y         xy'y         x'c?      x'xx'     xy'y         xy'y        xy'y           x'c?

so, these things really are?

xx'x + c x = 0

xx'xx' + c xx'= 0

xx' + c= 0

i think


'''

#########

x1,x2,b1,b2,y=sp.symbols("x1 x2 b1 b2 y")

obj = ( x1*b1+x2*b2 - y )**2

for var in [b1,b2]:
    diff = sp.diff(obj,var)
    print(var.name+":", diff.as_poly() )

#########
'''
(x*b - y)' (x*b - y)
(x*b)'(x*b) - (x*b)'y - y'(x*b) + y'y
b'x'x b - b'x'y - y'x*b + y'y
diff for b
x'x b - 2*x'y
'''
 
