import tensorflow as tf
import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

class GenerateData:
  data =[]
  processed_data =pd.DataFrame()
  def __init__(self, datafile_path,scaler_path,seq_length):
    GenerateData.data = pd.read_csv(datafile_path)
    GenerateData.data['layprice1'] = GenerateData.data['layprice1'].replace(0,1000)
    GenerateData.data['layprice2'] = GenerateData.data['layprice2'].replace(0,1000)
    GenerateData.data['layprice3'] = GenerateData.data['layprice3'].replace(0,1000)
    GenerateData.data['layprice4'] = GenerateData.data['layprice4'].replace(0,1000)
    GenerateData.data['layprice5'] = GenerateData.data['layprice5'].replace(0,1000)
    GenerateData.data['layprice6'] = GenerateData.data['layprice6'].replace(0,1000)
    GenerateData.data['layprice7'] = GenerateData.data['layprice7'].replace(0,1000)
    GenerateData.data['layprice8'] = GenerateData.data['layprice8'].replace(0,1000)
    GenerateData.data['layprice9'] = GenerateData.data['layprice9'].replace(0,1000)
    GenerateData.data['layprice10'] = GenerateData.data['layprice10'].replace(0,1000)

    GenerateData.data['laydepth1'] = GenerateData.data['laydepth1'].replace(0,1)
    GenerateData.data['laydepth2'] = GenerateData.data['laydepth2'].replace(0,1)
    GenerateData.data['laydepth3'] = GenerateData.data['laydepth3'].replace(0,1)
    GenerateData.data['laydepth4'] = GenerateData.data['laydepth4'].replace(0,1)
    GenerateData.data['laydepth5'] = GenerateData.data['laydepth5'].replace(0,1)
    GenerateData.data['laydepth6'] = GenerateData.data['laydepth6'].replace(0,1)
    GenerateData.data['laydepth7'] = GenerateData.data['laydepth7'].replace(0,1)
    GenerateData.data['laydepth8'] = GenerateData.data['laydepth8'].replace(0,1)
    GenerateData.data['laydepth9'] = GenerateData.data['laydepth9'].replace(0,1)
    GenerateData.data['laydepth10'] = GenerateData.data['laydepth10'].replace(0,1)

    GenerateData.data['backprice1'] = GenerateData.data['backprice1'].replace(0,1000)
    GenerateData.data['backprice2'] = GenerateData.data['backprice2'].replace(0,1000)
    GenerateData.data['backprice3'] = GenerateData.data['backprice3'].replace(0,1000)
    GenerateData.data['backprice4'] = GenerateData.data['backprice4'].replace(0,1000)
    GenerateData.data['backprice5'] = GenerateData.data['backprice5'].replace(0,1000)
    GenerateData.data['backprice6'] = GenerateData.data['backprice6'].replace(0,1000)
    GenerateData.data['backprice7'] = GenerateData.data['backprice7'].replace(0,1000)
    GenerateData.data['backprice8'] = GenerateData.data['backprice8'].replace(0,1000)
    GenerateData.data['backprice9'] = GenerateData.data['backprice9'].replace(0,1000)
    GenerateData.data['backprice10'] = GenerateData.data['backprice10'].replace(0,1000)

    GenerateData.data['backdepth1'] = GenerateData.data['backdepth1'].replace(0,1)
    GenerateData.data['backdepth2'] = GenerateData.data['backdepth2'].replace(0,1)
    GenerateData.data['backdepth3'] = GenerateData.data['backdepth3'].replace(0,1)
    GenerateData.data['backdepth4'] = GenerateData.data['backdepth4'].replace(0,1)
    GenerateData.data['backdepth5'] = GenerateData.data['backdepth5'].replace(0,1)
    GenerateData.data['backdepth6'] = GenerateData.data['backdepth6'].replace(0,1)
    GenerateData.data['backdepth7'] = GenerateData.data['backdepth7'].replace(0,1)
    GenerateData.data['backdepth8'] = GenerateData.data['backdepth8'].replace(0,1)
    GenerateData.data['backdepth9'] = GenerateData.data['backdepth9'].replace(0,1)
    GenerateData.data['backdepth10'] = GenerateData.data['backdepth10'].replace(0,1)

   # print(GenerateData.data)
    _nsequences = len(GenerateData.data) / seq_length
    #print(_nsequences)
    for i in range (_nsequences):
      _sequence = GenerateData.data[i * seq_length: (i * seq_length) + seq_length]
      #print(_sequence)

      diff = _sequence.pct_change()
      #print(diff)
      GenerateData.processed_data = GenerateData.processed_data.append(diff,ignore_index = True).fillna(value=0)
     
      #standardise everything


    scaler = StandardScaler()

    GenerateData.processed_data[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']]=scaler.fit_transform(GenerateData.processed_data[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']])
    #GenerateData.processed_data[['layprice1']] = scaler.fit_transform(GenerateData.processed_data[['layprice1']])
    #save the scaler for later use

    joblib.dump(scaler, scaler_path)

  def getTrainingSample(self,seq_length,batch_size, input_seq_length, output_seq_length):
    data = GenerateData.processed_data
    _nsequences = len(data) / seq_length
    input_batches = []
    output_batches = []

    for i in range(batch_size):
      #n_sequences = len(data) / (input_seq_length + output_seq_length)
      sequence_index = np.random.randint(0,_nsequences)
     # print(sequence_index)
      n_starting_indexes = seq_length - (input_seq_length + output_seq_length) -1
      #print(n_starting_indexes)
      starting_index = np.random.randint(0,n_starting_indexes)
      #print(starting_index)
      starting_offset = sequence_index * seq_length
      #print(starting_offset)
      an_input_sequence = data[starting_offset + starting_index: starting_offset + starting_index + input_seq_length]
      #print(an_input_sequence)

      input_data = an_input_sequence[['layprice1','laydepth1','layprice2','laydepth2','layprice3','laydepth3','layprice4','laydepth4','layprice5','laydepth5','layprice6','laydepth6','layprice7','laydepth7','layprice8','laydepth8','layprice9','laydepth9','layprice10','laydepth10','backprice1','backdepth1','backprice2','backdepth2','backprice3','backdepth3','backprice4','backdepth4','backprice5','backdepth5','backprice6','backdepth6','backprice7','backdepth7','backprice8','backdepth8','backprice9','backdepth9','backprice10','backdepth10']]
      input_batches.append(np.array(input_data))

      an_output_sequence = data[starting_offset + starting_index + input_seq_length:starting_offset +starting_index + input_seq_length + output_seq_length]
      #print(an_output_sequence)
      output_data = an_output_sequence[['layprice1','backprice1']]
      output_batches.append(np.array(output_data))


    return input_batches, output_batches

  def reshape(self, input_array, sequence_length, input_dimension):
    reshaped = [None]* sequence_length 

    for t in range(sequence_length):
      x = input_array[:,t].reshape(-1,input_dimension)
      reshaped[t]=x
    return(np.array(reshaped))

#test
gd = GenerateData('../nodejs/data/generate.csv','../nodejs/data/scaler.save',60)
#print("processed data: ")
#print(gd.processed_data)
#print(gd.data[0:60])
input_batch, output_batch = gd.getTrainingSample(60,2,30,10)
print(input_batch)
print(output_batch)

reshaped_input_batch = gd.reshape(np.array(input_batch),30, 40)
print(reshaped_input_batch)

reshaped_output_batch = gd.reshape(np.array(output_batch), 10, 2)
print(reshaped_output_batch)