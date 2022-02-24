# Classification of head and neck cancer from PET images using convolutional neural networks
This repository contains the code used to train and evaluate four different convolutional neural network models presented in the article "Classification of head and neck cancer from PET images using convolutional neural networks". The code is compiled into a single function for which it is easy to input training and test data. The function performs a number of training iterations using the given training data and finally evaluates the performance of the resulting model using the test data. The evaluation metrics are then finally saved as pkl-files which can be opened later for plotting and further analysis. 

## About the data format
The models expect the input data to be numpy-arrays with the shape (<number of instances>, <height>, <width>, <channels>). For example, in the experiments presented in the paper, the numebr of instances in the training data was 265. The instances were two-dimensional PET-images of size 128 x 128. Therefore the shape of the training data was (265, 128, 128, 1).

## About the pkl-files
The function creates a folder "results" if it does not already exists. In the folder, the function creates a pkl-file containing evaluation metrics for a single training iteraion. These files can later be accessed using the Python's dill-module:
```
import dill

with open('your_results.pkl', 'rb') as file:
    result_dictionary = append(pickle.load(file))
```

## Dependencies
The requirements needed to run the classification function are presented in the file 'requirements.txt'. Should any of these libraries be missing from the active Python installation, they can be installed in the command line (assuming pip is installed):
````
pip install -r /path/to/requirements.txt
````

## Sample usage
In this section we present a short example showing how this function has been used in the training of the models presented in the article. The number of epochs is constant throughout a single training routine and the number of epochs for each model configuration was determined using Tensorflow's <a href="https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping">EarlyStopping</a> functionality. For example, for the augmented deep model 35 epochs were used. In the example below, it is assumed the train and test data have been saved in the variables 'train_data' and 'test_data'. The same assumption applies for the variables 'train_labels' and 'test_labels'.
````
classify_images(
    train_data,
    train_labels,
    test_data,
    test_labels,
    50,
    35,
    deep=True
)
````
