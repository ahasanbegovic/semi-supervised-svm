from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

max_iter = 100

samples = 10000
# num_classes * num_clusters <= 2 ** num_informative
num_classes = 3
num_clusters = 2
num_informative = 3
ds = datasets.make_classification(n_samples=samples, n_informative=num_informative, n_classes=num_classes, n_clusters_per_class=num_clusters)
X = ds[0]
Y = ds[1]

seed = 7


svm = SVC()
scores = cross_val_score(svm, X, Y, cv=4)
print('SVM (fully labeled): %.2f'%scores.mean())
rng = np.random.RandomState(seed)

Y_wul = np.copy(Y)
Y_wul[rng.rand(len(Y)) < 0.98] = -1
start_labeled_data = len(np.array(Y_wul[Y_wul > -1]))
print('Labeled data at the start: ', start_labeled_data)

cost = 0

Y_small = np.array(Y_wul[Y_wul > -1])
X_small = np.array(X[Y_wul > -1])
svm2 = SVC(probability=True)
svm2.fit(X_small, Y_small)

scores2 = cross_val_score(svm2, X_small, Y_small, cv=4)
print('SVM (small labeled): %.2f'%scores2.mean())
improvement = 0
for k in range(0,max_iter):
    X_unlabeled = np.array(X[Y_wul == -1])
    svm2.fit(X_small, Y_small)
    Y_predictions = svm2.predict(X_unlabeled)
    Y_confidence = svm2.predict_proba(X_unlabeled)
    Y_confidence = np.amax(Y_confidence, axis=1)

    added = False
    for i in range(0, len(Y_confidence)):
        # Trust the labels the classifier is most confident about and include them in the training set
        if Y_confidence[i] >= 0.98:
            x = X_unlabeled[i,:]
            x = x[np.newaxis,:]
            original_index = np.where(X == x)[0][0]
            X_small = np.append(X_small, x, axis=0)
            Y_small = np.append(Y_small, Y_predictions[i])
            Y_wul[original_index] = Y_predictions[i]
            added = True
        # Get help from the human expert for the labels the classifier has the lowest confidence in
        if Y_confidence[i] <= 1/num_classes+0.03 and cost < start_labeled_data:
            x = X_unlabeled[i,:]
            x = x[np.newaxis,:]
            original_index = np.where(X == x)[0][0]
            X_small = np.append(X_small, x, axis=0)
            Y_small = np.append(Y_small, Y[original_index])
            Y_wul[original_index] = Y[original_index]
            added = True
            cost = cost + 1

    pseudo_scores = cross_val_score(svm2, X_small, Y_small, cv=4)
    print('SVM (with added pseudo-labels): %.2f'%pseudo_scores.mean())
    print('Cost: ', cost)
    improvement = (pseudo_scores.mean() - scores2.mean()) * 100
    if cost >= start_labeled_data:
        print('Maximum cost reached')
        break
    if pseudo_scores.mean() > 0.99:
        print('Accuracy reached 100%!')
        break
    if added is False:
        print('Accuracy cannot be further improved!')
        break
    if k == max_iter - 1:
        print('Maximum number of iterations achieved!')

print('Improvement: %.2f %%'%improvement)