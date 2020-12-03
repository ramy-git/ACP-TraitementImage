import numpy as np
import cv2
from numpy.linalg import svd, norm

# chargement de l'image
Ibgr = cv2.imread('lena.jpg')

Iycc = cv2.cvtColor(Ibgr, cv2.COLOR_BGR2YCR_CB)

# redimensionnement de la matrice mettons la sous la forme (N = W * H)
Izycc = Iycc.reshape([-1, 3]).T

#moyenne :  Pour que PCA fonctionne correctement, on soustrait la moyenne de chacune des dimensions.
Izycc = Izycc - Izycc.mean(1)[:, np.newaxis]
assert(np.allclose(np.mean(Izycc, 1), 0.0))

#on fait une décomposition singuliere c'est a dire . Il s’agit notamment de la reduction de la dimensionnalite, la compression d’image
(U, S) = svd(Izycc, full_matrices=False)[:2]


# tableau de 3 x 3 diagonales contenant les valeurs propres de la matrice de covariance 
eigvals = np.diag(S*2 / norm(S*2))

#vecteur propres
eigvecs = U;

#transposition 
Igray = np.dot(eigvecs.T, np.dot(eigvals, Izycc)).sum(0).reshape(Iycc.shape[:2])

# Redimensionner Igray à [0, 255].en arrondissant en entier en faisant une interpolation et une reduction
from scipy.interpolate import interp1d
Igray = np.floor((interp1d([Igray.min(), Igray.max()],
                            [0.0, 256.0 - 1e-4]))(Igray))


if norm(Iycc[:,:,0] - Igray) > norm(Iycc[:,:,0] - (255.0 - Igray)):
    Igray = 255 - Igray

on affiche les resultats
if True:
    import pylab
    pylab.ion()
    pylab.imshow(Igray, cmap='gray')

# On enregistre ensuite l'image dans le dossier courant
cv2.imwrite('lena.png', Igray.astype(np.uint8))