# Group 14

- Michael Etschbacher
- Raphael Schotola
- Gabriel Breiner
- Oliver Stritzel

## Toolkit

- PySpark
- JupyterNotebook
- pandas
- scipy
- numpy
- pytorch

## Preprocessing

`twitter_preproc.py` inherits the code for preprocessing the raw data. This is a class which is re-used by several others, with different approaches for preprocessing (f.i. random forest needs all features, matric factorization only want the id’s)

## Entry Points to our Approaches

### Neural Network Collaborative Filtering (NNCF)

`nnpreprocessing.py` contains nncf-specific preprocessing (one-hot-encoding)
`NNCFNet.py` contains the neural network class
`nncf_notebook.ipynb` portrays our approach for NNCF using the 2 classes above on a split of the training set
`spark-submit nncf_submit.py` trains our network on the full training set

###  Random Forest

`forest.py` can be run via spark-submit which will yield a training and classification of a random forest classifier.

### Content Based Approach

This approach runs on the jupyter notebook “src/content_based.ipynb” as a standalone notebook, change the path specs if necessary
