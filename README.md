# __End2End Dialogue__
End-to-End Dialogue Management (on DSTC2)
___
## Author
* __Qi Hu__ Email: qihuchn@gmail.com

___
## Function
 * Dialogue Act Prediction and Language Generation Models
   1. Classification Model (CNN) to predict the system's response to user
   2. Language Model (LSTM) to create system response according to dialogue act and slot
 * Merge the above two models to get an End-to-End model without explicit dialogue act and slot
## Algorithm
CNN Encoder + LSTM Decoder based on the paper "A Network-based End-to-End Trainable Task-oriented Dialogue System"
___

## Dataset
DSTC2 (dialog state tracking challenge) dataset
3225 dialogs and average length of dialog is 14 turns. Each pair includes the user query asn system response.

The dataset is stored in several files in directory "data".

## Usage
#### Basic Model
  1. __*python e2e_dstc.py 0 0*__: Train e2e model from scratch
  2. __*python e2e_dstc.py 0 0*__: Continue the training from latest checkpoint 
  3. __*python e2e_dstc.py 1 0*__: Test e2e model

#### Attention Model (buggy)
  1. __*python e2e_dstc_attention.py 0 0*__: Train e2e model from scratch
  2. __*python e2e_dstc_attention.py 0 0*__: Continue the training from latest checkpoint 
  3. __*python e2e_dstc_attention.py 1 0*__: Test e2e model

## Other Files
1. __tool/data_loader.py__
   Reading training and testing data from files
2. __tool/data_processor.py__
   Pre-processing raw data
