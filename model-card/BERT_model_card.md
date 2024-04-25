---
{}
---
language: en
license: cc-by-4.0
tags:
- Natural Language Inference
repo: 'https://github.com/ShuoYin03/COMP34812_NLI'

---

# Model Card for Qian Yang-Shuo Yin-NLI

<!-- Provide a quick summary of what the model is/does. -->
The model is designed to predict binary outcomes based on the relationship between pairs of text sequences, typically a premise and a hypothesis.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a BERT model that was fine-tuned on 30K pairs of texts, followed by dense layers to extract features and output with a sigmoid layer.

- **Developed by:** Qian yang and Shuo Yin
- **Language(s):** python
- **Model type:** Supervised
- **Model architecture:**  BERT (Bidirectional Encoder Representations from Transformers)
- **Finetuned from model [optional]:** small bert uncased L-4 H-256 A-4

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1'
- **Paper or documentation:** https://www.kaggle.com/models/tensorflow/bert/tensorFlow2/bert-en-uncased-l-4-h-256-a-4

## Training Details
### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

 26K premise-hypothesis pairs 

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

      - learning_rate: 0.00002
      - train_batch_size: 16
      - val_batch_size: 16
      - num_epochs: 3

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->

      - overall training time: 1h 46m 33s
      - duration per training epoch: 2131s
      - model size: 42.61 MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->
### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

3.3K premise-hypothesis pairs

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - F1 Score

### Results

The model obtained an F1-score of 77.04% and an accuracy of 76.81%.

## Technical Specifications

### Hardware

      - RAM: at least 12 GB
      - Storage: at least 2GB,
      - GPU: Tesla T4

### Software

      - Transformers 4.40.0
      - Tensorflow 2.10.1
      - Tensorflow_hub 0.16.1
      - Tensorflow-text 2.13
      - NumPy 1.26.4
      - Pandas 2.2.2

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

If the input text connecting two sequences exceeds 20 subwords, the excess will be truncated

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The model is fine-tuned during training, as indicated by trainable=True when loading the BERT encoder layer.
