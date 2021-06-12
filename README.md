# Kaggle Titanic Classification using KNN

A simple Python programming classification using KNN model to classify Kaggle Titanic Problem [link](https://www.kaggle.com/c/titanic/). 
The final score submission is 0.77751.

## Content
- [x] Python Script (knn_titanic_model.py)
- [x] Readme
- [ ] Notebook (later_project)

## Dataset

The training set and test set could be downloaded from Kaggle [link](https://www.kaggle.com/c/titanic/data).

## The Methodology Used
### Preprocessing
* using Median for missing Age
* using the median of inclusive Pclass for the missing fare
* Normalized using Standart Scaler
### Validation
* stratifiedkfold (n=5)
### Classification Model
* KNN (k=5)

## Acknowledge
>  self project by Kevin Kristian