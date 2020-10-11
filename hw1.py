# import import_ipynb
from midi2seq import *
from model_base import *
import torch
import torch.nn as nn
import zipfile
import io
# import torchnlp.download
from torch.utils.data import *
#import torch.nn.functional as F
#import torch. optim as optim

#torch.manual_seed(1)

class Composer(ComposerBase):
    def __init__(self,load_trained=False):
        self.load_trained=load_trained
#         self.LSTM=None
        print("this is the Composer constructor.")
    
    def train(self,x):
#         print("this is train method in Composer class.")
#         print(self.load_trained)
#         url='https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip'
#         a=torchnlp.download.download_file_maybe_extract(url, './data')
        print('----------------------------------------------------------------------------')
        print(x)
        if self.load_trained==True:
            pass
        if self.load_trained==False:
            print("We are inside the conditional.")
        
            
            embedding = nn.Embedding(520, 520)
            print(embedding)
            linear=nn.Linear(520,520)
            
            inputs=embedding(x)
            print(inputs)
            
            lstm=nn.LSTM(input_size=51,hidden_size=51,num_layers=2)
            
            
            hidden = (torch.randn(2, 1, 51), torch.randn(2, 1, 51))  # clean out hidden state
            
            out, hidden = lstm(inputs, hidden)
            out=linear(out)
            
#             self.LSTM=lstm
            
#             print(out)
#             print(hidden)
        


#             return seq2piano(out)
        
        print(out)
        print(hidden)
    
    def compose(self,n=100):
        print("This is the compose method.")
        print(self.load_trained)
        return 1
        

class Critic(CriticBase):
    
    def __init__(self,load_trained=False):
        print("this is the Critic constructor.")
        self.load_trained=load_trained
    
    def train(self):
        print("this is train method in Critic class.")
        print(self.load_trained)
        return 1
    
    def score(self):
        
        # Good music
        good_music=torch.from_numpy(process_midi_seq(all_midis=None, datadir='./data/good', n=10000, maxlen=50))
        train_data_good=DataLoader(TensorDataset(good_music), shuffle=True, batch_size=bsz, num_workers=4)
#         train_data_good_lbl=(train_data_good,'True')
        
        # Bad music
        # import a sequence using random_piano function in midi2seq as 
        # a bad example
        bad_midi=[]
        for i in range(10000):
            temp_midi=random_piano()
            bad_midi.append(temp_midi)
            
        bad_music=torch.from_numpy(process_midi_seq(all_midis=bad_midi))
        train_data_bad=DataLoader(TensorDataset(bad_music), shuffle=True, batch_size=bsz, num_workers=4)
        
            
#         train_data_bad=piano2seq(random_piano(50))
#         train_data_bad_lbl=(train_data_bad,'False')
        
        # Create a 2-3 layer LSTM model (Seach how to create LSTM in PyTorch)
        # input_size is the same as PyTorch 
        lstm=nn.LSTM(input_size=51,hidden_size=1,num_leyers=2)    # Input dim=1 (sequence of events) Output dim= 2 (label)
        #inputs=[torch.randn(1,3) for _ in range(5)]
        
        
#         hidden = (torch.randn(1, 1, 3),
#           torch.randn(1, 1, 3))
        
        
        
        # import a sequence using downloaded folder as a good example
        
        
        
        # Implement the learning methode here to train the network 
        # in order to make distinction between good and bad examples
        
        