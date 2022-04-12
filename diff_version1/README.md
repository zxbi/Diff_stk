## Dependencies
The code was written in `Python 3.6.4` and requires `PyTorch 1.2.0` and a couple of other dependencies. 
To install them run the following command
```
pip install -r requirements.txt
```

## Task and Model Parameters
A probabilistic context-free grammar for Dyck-n can be written as follows:

Refer paper or the folder with name visualization

where 0 < p, q < 1 and p + q < 1.

In our code, we represent the paremeters `n`, `p`, and `q` in the above formulation by `num_par`, `p_val`, and `q_val`, respectively:
* `num_par`: Number of parentheses pairs.
* `p_val`: p value.
* `q_val`: q value.

We can further specify the number of samples in the training and test corpura:
* `training_window`: Training set length window.
* `training_size`: Number of samples in the training set.
* `test_size`: Number of samples in the test set.

A single layer DiffStack-RNN (`stack_rnn_softmax`) with 8 hidden units and 5 dimensional memory is chosen as a default architecture, however we can easily change the model parameters with the following set of arguments:
* `model_name`: Model choice (e.g., `stack_rnn_softmax` for DiffStack-RNN, `baby_ntm_softmax` for Baby-NTM (untested and unstable version)).
* `n_layers`: Number of hidden layers in the network.
* `n_hidden`: Number of hidden units per layer in the network.
* `memory_size`: Size of the stack/memory.
* `memory_dim`: Dimensionality of the stack/memory.


Learning rate and number of epochs can be specified with the following parameters:
* `lr`: Learning rate.
* `n_epoch`: Number of epochs to be trained.

Finally, we can save and load the model weights by using the following arguments. By default, if `load_path` is not specified, the code trains a model from scratch and then saves the model weights in the `models` folder.
* `save_path`: Path to save the weights of a model after the trainig is completed.
* `load_path`: Path to load the weights of a pre-trained model.


## Training
To train a single layer DiffStack-RNN model with 8 hidden units and 1-dimensional stack to learn the Dyck-2 language:

`python main.py --num_par 2 --model_name stack_rnn_softmax --n_hidden 8 --memory_dim 5 --save_path models/stack_rnn_model_weights.pth`

To train a single layer Diff-Baby Neural Turing Machine (Baby-NTM) with 8 hidden units and 5-dimensional memory to learn the Dyck-3 language:

`python main.py --num_par 3 --model_name baby_ntm_softmax --n_hidden 8 --memory_dim 5 --save_path models/baby_ntm_model_weights.pth`

## Evaluation
To evaluate the performance of the Diff-Baby-NTM model trained in the previous section, we can simply write:

`python main.py --num_par 3 --model_name baby_ntm_softmax --n_hidden 8 --memory_dim 5 --load_path models/baby_ntm_model_weights.pth`


## Visualizations
Look at all notebooks to understand this work.

## Things to do
Increase training sequence length, create a validation set (max is train represents maximum length of language in other words N, so a^nb^n, here n = max). For validation have data of length max+2, max+15, modify test to have symbols max+17 and max+50. Perform experiments with all architectures

## Important 
Do not share this code beside your teammates, sharing this can lead to IP issues.