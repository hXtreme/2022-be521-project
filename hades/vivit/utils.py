import pickle

def load_batch(data_path, data_split, batch_num):
  addr = data_path + data_split + '/' + str(batch_num) + '.pkl'
  with open(addr, 'rb') as infile:
      data = pickle.load(infile)
  return data[1], data[0]

def load_audio_batch(audio_data_path, data_split, batch_num):
  addr = audio_data_path + data_split + '/' + str(batch_num) + '.pkl'
  with open(addr, 'rb') as infile:
      data = pickle.load(infile)
  return data[1], data[0]