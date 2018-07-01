# semi-supervised-svm

Data science assignment solution. Implementation of a semi-supervised classifier using Support Vector Machines as the base classifier. The dataset is randomly generated in the code.

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

## Solution
The solution adds the predicted labels with the highest confidence to the labeled dataset. The labels with the lowest confidence show the classifier needs help form the human expert. Those true labels are added to the dataset and the cost is incremented. The amount of hints from the human expert cannot exceed the initial number of labeled samples - the amount of labeled data can only be doubled. 

## Example

Setup:
- Dataset: 10000 samples, 3 classes, 2 clusters per class, 3 informative features.
- Max. number of iterations: 100
- Percentage of unlabeled data in the dataset: 98%
- Min. confidence at which the predicted label is considered accurate: 98%
- Max. confidence at which the human expert will be asked to replace a bad label: 36.33%

Output:
```
SVM (fully labeled): 0.82
Labeled data at the start:  192
SVM (small labeled): 0.66
SVM (with added pseudo-labels): 0.60
Cost:  46
SVM (with added pseudo-labels): 0.75
Cost:  60
SVM (with added pseudo-labels): 0.75
Cost:  74
SVM (with added pseudo-labels): 0.75
Cost:  82
SVM (with added pseudo-labels): 0.78
Cost:  91
SVM (with added pseudo-labels): 0.77
Cost:  101
SVM (with added pseudo-labels): 0.76
Cost:  109
SVM (with added pseudo-labels): 0.77
Cost:  111
SVM (with added pseudo-labels): 0.76
Cost:  115
SVM (with added pseudo-labels): 0.75
Cost:  117
SVM (with added pseudo-labels): 0.75
Cost:  118
SVM (with added pseudo-labels): 0.76
Cost:  122
SVM (with added pseudo-labels): 0.77
Cost:  123
SVM (with added pseudo-labels): 0.77
Cost:  126
SVM (with added pseudo-labels): 0.78
Cost:  127
SVM (with added pseudo-labels): 0.77
Cost:  128
SVM (with added pseudo-labels): 0.77
Cost:  128
SVM (with added pseudo-labels): 0.77
Cost:  128
Accuracy cannot be further improved!
Improvement: 11.32 %
```
