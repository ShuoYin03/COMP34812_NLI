---
{}
---
language: en
license: cc-by-4.0
tags:
- Natural Language Inference
repo: https://github.com/ShuoYin03/COMP34812_NLI

---

# Model Card for Shuo_Yin-Qian_Yang-NLI

<!-- Provide a quick summary of what the model is/does. -->

This model is designed to determine the relationship between two sentences (premise and hypothesis) 
    by classifying them as entailment or contradiction.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The model employs Bi-LSTM layers to capture contextual information from both forward and backward directions for each premise and hypothesis, and the result of Bi-LSTM layers are combined and processed by dense layer with tanh activation function to further extract the features.

- **Developed by:** Shuo Yin, Qian Yang

- **Language(s):** Python

- **Model type:** Supervised
- **Model architecture:** Bidirectional LSTM (Bi-LSTM) with FastText embedding layer
- **Finetuned from model [optional]:** Not applicable

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** N/A
- **Paper or documentation:** N/A

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

The model was trained on a dataset similar to the SNLI (Stanford Natural Language Inference) corpus, which consists of sentence pairs annotated with labels of entailment, contradiction, or neutral.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

Learning Rate: 0.0001  
Epochs: 7
Batch Size: 32
Optimizer: Adam  

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 7m 21s
      - duration per training epoch: 73s
      - model size: 14.18 MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

26k pairs of premise and hypothesis.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - F1 Score

### Results

The model obtained an accuracy of 67.89% and an F1 score of 70.45%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - CPU: Intel(R) Xeon(R) CPU @ 2.20GHz

### Software


      - NLTK 3.8.1
      - Genism 4.3.2
      - Tensorflow 2.15.1

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any input longer than 15 will be truncated.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyper-parameters were determined by experimentation
      with different values. Due to the small dataset, we had a small model to process the data
