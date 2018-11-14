import numpy as np
import imageio
from os import listdir
from time import time

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

np.random.seed(1)

path = './Asian'

h = 250
w = 250
D = h * w
N = 900 # the number of images
K = 100

############################################################################
def rgb2gray(rgb):
    #Y' = 0.299*R + 0.587*G + 0.114*B
    return rgb[:,:,0]*.299 + rgb[:,:,1]*.587 + rgb[:,:,2]*.114

# feature extraction
def vectorize_img(filename):
    # load image
    rgb = imageio.imread(filename)
    #convert to gray scale
    gray = rgb2gray(rgb)
    # vectorization each row is a data point
    im_vec = gray.reshape(1, D)
    return im_vec

def build_data_matrix(file_num):
    """
    INPUT:
        file_num: number of images
    """
    print('build data matrix ...')
    cnt = 0
    X = np.zeros((D, N))
    y = np.zeros((N, ))
    for fn in listdir(path):
        if(cnt < file_num):
            if (fn.find('R.') != -1) or (fn.find('L.') != -1):
                X[:, cnt] = vectorize_img(path + '/' + fn)
                if fn.find('R.') != -1:
                    y[cnt] = 1
                cnt += 1

    print('done.')
    return (X, y)

(X, y) = build_data_matrix(N)

(X_train, X_test, y_train, y_test) = train_test_split(X.T, y, test_size=0.222, random_state=42)

print('Decomposition...')
pca = PCA(n_components=K)
pca.fit(X_train)
print('Done.')

##########################################
# # projection matrix
# U = pca.components_.T

# # check
# import matplotlib.pyplot as plt
# for i in range(U.shape[1]):
#     plt.axis('off')
#     f1 = plt.imshow(U[:, i].reshape(w, h), interpolation='nearest')
#     f1.axes.get_xaxis().set_visible(False)
#     f1.axes.get_yaxis().set_visible(False)
#     # f2 = plt.imshow(, interpolation='nearest' )
#     plt.gray()
#     # fn = 'eigenface' + str(i).zfill(2) + '.png'
#     # plt.savefig(fn, bbox_inches='tight', pad_inches=0)
#     plt.show()

##########################################

print('Transform...')
eigenfaces = pca.components_.reshape((K, h, w))
t0 = time()
X_train_pca = pca.transform(X_train)    
X_test_pca = pca.transform(X_test)
print('Done in %0.3fs' % (time() - t0))

##########################################
# try Logistic Regression
logreg = linear_model.LogisticRegression(C=1e9)

logreg.fit(X_train_pca, y_train)
y_pred = logreg.predict(X_test_pca)
print('Accuracy for logistic regression: %.2f %%' %(100 * accuracy_score(y_test, y_pred)))

#########################
# try SVC
print('\nMore SVC')
svr_lin = SVC(C=1e3, kernel='linear')
svr_poly = SVC(C=1e3, kernel='poly', degree=2)
svr_rbf = SVC(C=1e3, kernel='rbf', degree = 3, gamma=1)

svr_lin.fit(X_train_pca, y_train)
y_svc_lin_pred = svr_lin.predict(X_test_pca)
print('Accuracy for linear: %.2f %%' %(100 * accuracy_score(y_test, y_svc_lin_pred)))

svr_poly.fit(X_train_pca, y_train)
y_svr_poly_pred = svr_poly.predict(X_test_pca)
print('Accuracy for poly: %.2f %%' %(100 * accuracy_score(y_test, y_svr_poly_pred)))

svr_rbf.fit(X_train_pca, y_train)
y_svr_rbf_pred = svr_rbf.predict(X_test_pca)
print('Accuracy for rbf: %.2f %%' %(100 * accuracy_score(y_test, y_svr_rbf_pred)))
