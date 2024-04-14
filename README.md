# SEResNet68
## 📦 Overview
We came up with a modified ResNet architecture with the highest test accuracy of 96.48% on the
CIFAR-10 image classification dataset, under the constraint
of no more than 5 million trainable parameters.

## ⚙️ Prerequisites

- Python 3.8.8
- torch 1.10.0+cu113
- torchvision 0.11.1+cu113
- pytorch_optimizer
- numpy
- pandas
- collections

## 🏁 Description of Files in the Repo

- Model_Weights_and_Eval_Metrics/ : Model weights trained with different hyperparameters and loss&acc for each epoch of training and testing.
- plots/ : Visualize acc and loss of models trained and tested using different hyperparameters in each epoch
- SE_ResNet_55.py : SEResNet model with 55 layers.
- SE_ResNet_68.py : SEResNet model with 68 layers.
- cifar_test_nolabels.pkl : A custom test dataset.
- inference_on_kaggle.ipynb : Inference notebook using the highest accuracy model to predict and generate submission on Kaggle.
- main.ipynb : Train and test.
- submission.csv : Prediction of the custom test dataset and submission to submit on Kaggle.

## ⏳ Training and Testing
Run `main.ipynb` to reproduce the result.
You need to modify different hyperparameters and select different network SEResNet architectures in `main.ipynb` to conduct different experiments.


## 📊 Results
| Sr. No. | Model Name   | # Residual Blocks in Residual Layer | Optimizer       | lr   | Augmentation | Gradient Clip | Batch Size | Params | Test Acc | File Link                                                                                                                                                         |
|---------|--------------|-------------------------------------|-----------------|------|--------------|---------------|------------|--------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1       | SEResnet55   | [2,2,2,2]                           | Lookahead+SGD   | 0.1  | True         | True          | 32         | 4.99M  | 95.81%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/4residual_layers_model)                                       |
| 2       | SEResnet68   | [4,4,3]                             | Lookahead+SGD          | 0.1  | True         | True          | 32         | 4.70M  | 96.28%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/batch_size32_model)                                           |
| **3**   | **SEResnet68** | **[4,4,3]**                       | **Ranger** | **0.1** | **True**     | **True**      | **128**      | **4.70M** | **96.48%** | [**LINK**](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/best_acc_model)                                          |
| 4       | SEResnet68   | [4,4,3]                             | Lookahead+SGD   | 0.01 | True         | True          | 32         | 4.70M  | 96.23%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/lr0.01_model)                                                 |
| 5       | SEResnet68   | [4,4,3]                             | Lookahead+SGD   | 0.1  | True         | True          | 32         | 4.70M  | 95.67%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/sgd_model)                                                    |
| 6       | SEResnet68   | [4,4,3]                             | Lookahead+SGD   | 0.1  | False        | True          | 32         | 4.70M  | 91.82%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/without_aug_model)                                            |
| 7       | SEResnet68   | [4,4,3]                             | Lookahead+SGD   | 0.1  | True         | False         | 32         | 4.70M  | 95.80%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/without_gradient_model)                                      |


## 👩‍⚖️ Acknowledgement
*Authors:* Shengyang(Steven Li), Xinyan Xie, Sitong Chen 
