# Kernel-SVM
Implemented Kernel SVM using Quadratic Programming and Stochastic Gradient Descent
In this project, SVMs are implemented using two different optimization techniques: 
1. Quadratic Programming
2. Stochastic Gradient Descent

## Kernel SVM using Quadratic Programming
Quadratic programs refer to optimization problems in which the objective function is quadratic and the constraints are linear. Quadratic programs are well studied in optimization literature, and there are efficient solvers. Many Machine Learning algorithms are reduced to solving quadratic programs. In this project, we will use the quadratic program solver of Matlab to optimize the dual objective of a kernel SVM.

## Kernel SVM using Stochastic Gradient Descent
In this project, mutliclass SVM is implemented with Stochastic Gradient Descent. Crammer and Singer’s SVM formulation is considered for implmenting SVM, `https://www.csie.ntu.edu.tw/ ̃cjlin/papers/multisvm.pdf`. Note that there are several different formulations for multiclass

## Dataset
Applied Kernel SVM on UCF101 `http://crcv.ucf.edu/data/UCF101.php` dataset but with only 10 classes. The feature vectors of activities of the 10 classes are already extracted. Each feature vector has 4096 features. The training data is provided here `http://bit.ly/2HyRsLS`.

Training data is provided in q3_2_data.mat. Use trD, trLb for training your SVM classifier. Validate your obtained SVM on valD, valLb, then provide the prediction for tstD in a .csv file.

For reference, the jpeg images from which the feature vectors were extracted is also provided. The training and validation labels are correspondence to trLb and valLb in q3_2_data.mat. 

## Kaggle Competition
Activity Recognition using Kernel SVM - Ranked 8th - `https://www.kaggle.com/c/hw2-activity-recognition-cse512-spr18/`
