# semi-supervised-svm

Data science assignment solution. Implementation of a semi-supervised classifier using Support Vector Machines as the base classifier.

## Dependencies:
- numpy
- sklearn

## Classification problem
Given data:
- a large amount of unlabeled data
- a small amout of labeled data
- a human expert able to correctly label any sample in the unlabeled dataset for a cost proportional to the number of newly labeled samples

Goal:
- minimize the cost
- improve the accuracy of the classifier
