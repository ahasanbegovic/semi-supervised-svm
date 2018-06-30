from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.svm import SVC
import numpy as np

max_iter = 10

samples = 10000
# num_classes * num_clusters <= 2 ** num_informative
num_classes = 3
num_clusters = 2
num_informative = 3
ds = datasets.make_classification(n_samples=samples, n_informative=num_informative, n_classes=num_classes, n_clusters_per_class=num_clusters)
X = ds[0]
Y = ds[1]

validation_size = 0.10
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state = seed)


svm = SVC()
svm.fit(X_train, Y_train)

predictions_best = svm.predict(X_validation)
svm_accuracy = accuracy_score(Y_validation, predictions_best)
print('SVM: ', svm_accuracy)
rng = np.random.RandomState(seed)


Y_train_wul = np.copy(Y_train)
Y_train_wul[rng.rand(len(Y_train)) < 0.98] = -1
start_labeled_data = len(np.array(Y_train_wul[Y_train_wul > -1]))
print('Labeled data at the start: ', start_labeled_data)

cost = 0

Y_train_small = np.array(Y_train_wul[Y_train_wul > -1])
X_train_small = np.array(X_train[Y_train_wul > -1])
svm2 = SVC(probability=True)
svm2.fit(X_train_small, Y_train_small)

predictions_small = svm2.predict(X_validation)
svm2_accuracy = accuracy_score(Y_validation, predictions_small)
print('SVMsmall: ', svm2_accuracy)
improvement = 0
for k in range(0,max_iter):
    X_unlabeled = np.array(X_train[Y_train_wul == -1])
    Y_predictions = svm2.predict(X_unlabeled)
    Y_confidence = svm2.predict_proba(X_unlabeled)
    Y_confidence = np.amax(Y_confidence, axis=1)

    added = False
    for i in range(0, len(Y_confidence)):
        # Trust the ones the classifier is most confident about and include them in the training set
        if Y_confidence[i] >= 0.98:
            x = X_unlabeled[i,:]
            x = x[np.newaxis,:]
            original_index = np.where(X_train == x)[0][0]
            X_train_small = np.append(X_train_small, x, axis=0)
            Y_train_small = np.append(Y_train_small, Y_predictions[i])
            Y_train_wul[original_index] = Y_predictions[i]
            added = True
        # Get help from the human expert for the ones the classifier has the lowest confidence
        if Y_confidence[i] <= 1/num_classes+0.04 and cost < start_labeled_data:
            x = X_unlabeled[i,:]
            x = x[np.newaxis,:]
            original_index = np.where(X_train == x)[0][0]
            X_train_small = np.append(X_train_small, x, axis=0)
            Y_train_small = np.append(Y_train_small, Y_train[original_index])
            Y_train_wul[original_index] = Y_train[original_index]
            added = True
            cost = cost + 1

    svm2.fit(X_train_small, Y_train_small)
    predictions_pseudo = svm2.predict(X_validation)
    svmPseudo_accuracy = accuracy_score(Y_validation, predictions_pseudo)
    print('SVMpseudo: ', svmPseudo_accuracy)
    print('Cost: ', cost)
    improvement = (svmPseudo_accuracy - svm2_accuracy) * 100
    if cost >= start_labeled_data:
        print('Maximum cost reached')
        break
    if svmPseudo_accuracy >= svm_accuracy - 0.05:
        print('Accuracy close to fully labeled results!')
        break
    if added is False:
        print('Accuracy can not be further improved!')
        break
    if k == max_iter - 1:
        print('Maximum number of iterations achieved')
        
print('Improvement: %.2f %%'%improvement)