# __End2End Dialogue__
End-to-End Dialogue Management
___
## Author
* __Qi Hu__ Email: qihu@mobvoi.com

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
420 pairs of dialogue. Each pair includes the user query, VD feature of user query, current dialogue state, 
dialogue act, dialogue slot and system response

There are 14 dialogue acts: inform, offerrestaurant, request, suggest, noresult, affirm, confirm, negate, goodbye,
requestalts, sorry, thankyou, youarewelcome, other. 6 slots: area, score, name, price, address, food_type. Only the 
first 5 acts has slot information.

The dataset is stored in several files in directory "data". The purpose of splitting data into several parts is to
make it easier to modify.

## Usage
#### End-to-End Model
  1. __*python e2e.py 0*__: Train e2e model
  2. __*python e2e.py 1*__: Test e2e model

#### No End-to-End Model (Just for test)
  1. __*python no_e2e.py*__: Test no_e2e model (Need to run __python da.py 0__ and __python lg_beam.py 0__
   first to train the model)


#### Encoder (CNN based)
  1. __*python da.py 0*__: Train e2e model
  2. __*python da.py 1*__: Test e2e model

#### Decoder (LSTM)
  1. __*python lg.py 0*__: Train e2e model
  2. __*python lg.py 1*__: Test e2e model

The results of All LG models will be stored in corresponding tmp directories.
(For instant, result of lg_beam is in tmp/lg_beam/answer.txt)

## Other Files
1. __tool/data_loader.py__
   Reading training and testing data from files
2. __tool/data_processer.py__
   Pre-processing raw (json) data collected from online WoZ system
3. __tool/eval_helper.py__
   Define functions for evaluating
4. __tool/foo.py__
   Get statistics of collected data
5. __tool/vds_caller.py & tool/cache_caller.py__
   Helpers to call online query-analysis service

