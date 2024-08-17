import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from pytorch_utils import do_mixup, interpolate, pad_framewise_output
from quaternion_layers import QuaternionConv2d ,QuaternionConv2dNoBias, QuaternionConv, QuaternionLinear
from torchinfo import summary
import torchaudio


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class QConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(QConvBlock, self).__init__()
        
        self.conv1 = QuaternionConv2dNoBias(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = QuaternionConv2dNoBias(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        # init_layer(self.conv1)
        # init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class QConvBlock_pruned(nn.Module):
    def __init__(self, in_channels_1, out_channels_1,out_channels_2):
       
        super(QConvBlock_pruned, self).__init__()
       
        self.conv1 = QuaternionConv2dNoBias(in_channels=in_channels_1, 
                              out_channels=out_channels_1,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
                             
        self.conv2 = QuaternionConv2dNoBias(in_channels=out_channels_1, 
                              out_channels=out_channels_2,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
                             
        self.bn1 = nn.BatchNorm2d(out_channels_1)
        self.bn2 = nn.BatchNorm2d(out_channels_2)

        self.init_weight()
       
    def init_weight(self):
        # init_layer(self.conv1)
        # init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

       
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
       
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
       
        return x
    
class QConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(QConvBlock5x5, self).__init__()
        
        self.conv1 = QuaternionConv2dNoBias(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        # init_layer(self.conv1)
        init_bn(self.bn1)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x





class QCNN14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(QCNN14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = QConvBlock(in_channels=4, out_channels=64)
        self.conv_block2 = QConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = QConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = QConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = QConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = QConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = QuaternionLinear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        # init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        print("input shape", input.shape)
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        
        
        
        x_first = torchaudio.functional.compute_deltas(x)
        x_second = torchaudio.functional.compute_deltas(x_first)
        x_third = torchaudio.functional.compute_deltas(x_second) 
        #quternionic converter
        x_quaternion = torch.cat([x,x_first, x_second, x_third], dim=1) # (batch_size, 4, time_steps, freq_bins)
        #print("spectrogram shape", x_quaternion.shape)
        x = self.logmel_extractor(x_quaternion)    # (batch_size, 4, time_steps, mel_bins)
        #print("logmel shape", x.shape)

        x = x.transpose(1, 3)
        #print("logmel shape after first transpose", x.shape)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        #print(" Shape after bachnorm normalisation transpose", x.shape)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
    
# model  = QCnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,fmin=50,fmax=14000,classes_num=528)
# # print(model(torch.rand(1,10*32000)))
# summary(model, (1,10*32000))

# Total params: 22,097,936
# Trainable params: 21,014,480
# Non-trainable params: 1,083,456
# Total mult-adds (G): 36.33
# ==========================================================================================
# Input size (MB): 1.28
# Forward/backward pass size (MB): 270.42
# Params size (MB): 88.39
# Estimated Total Size (MB): 360.09

class QCnn14_pruned_25(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
            fmax, classes_num):#, pooling_type, pooling_factor):
        import os
        import numpy as np

        super(QCnn14_pruned_25, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        from collections import OrderedDict
        path = '~/audio_tagging_quaternion-main/sorted_index_L1_pruning/' #'/home/arshdeep/Pruning/PANNs_pruning/sorted_Indexes/OP/'   

        p = 0.25
        p1 = 0
        p2 = 0
        p3 = 0
        p4 = 0
        p5 = 0
        p6 = 0
        p7 = p
        p8 = p
        p9 = p
        p10 = p
        p11 = p
        p12 = p

        C1_r = np.arange(0,16)[int(16*p1):16]#sorted(np.load(os.path.join(path,'conv_block1.conv1.weight.npy'))[int(64*p1):64])
        C1_i = np.arange(0,16)[int(16*p1):16]
        C1_j = np.arange(0,16)[int(16*p1):16]
        C1_k = np.arange(0,16)[int(16*p1):16]
        # C1_k = np.arange(0,100)[16+int(16*p1):32]
        # 
        C2_r= np.arange(0,16)[int(16*p2):16]#sorted(np.load(os.path.join(path,'conv_block1.conv2.weight.npy'))[int(64*p2):64])
        C2_i= np.arange(0,16)[int(16*p2):16]
        C2_j= np.arange(0,16)[int(16*p2):16]
        C2_k= np.arange(0,16)[int(16*p2):16]

        C3_r = np.arange(0,32)[int(32*p3):32]
        C3_i = np.arange(0,32)[int(32*p3):32]
        C3_j = np.arange(0,32)[int(32*p3):32]
        C3_k = np.arange(0,32)[int(32*p3):32]


        #sorted(np.load(os.path.join(path,'conv_block2.conv1.weight.npy'))[int(128*p3):128])
        C4_r = np.arange(0,32)[int(32*p4):32]
        C4_i = np.arange(0,32)[int(32*p4):32]
        C4_j = np.arange(0,32)[int(32*p4):32]
        C4_k = np.arange(0,32)[int(32*p4):32]


        #sorted(np.load(os.path.join(path,'conv_block2.conv2.weight.npy'))[int(128*p4):128])


        C5_r = np.arange(0,64)[int(64*p5):64]
        C5_i = np.arange(0,64)[int(64*p5):64]
        C5_j = np.arange(0,64)[int(64*p5):64]
        C5_k = np.arange(0,64)[int(64*p5):64]

        #sorted(np.load(os.path.join(path,'conv_block3.conv1.weight.npy'))[int(256*p5):256])
        C6_r = np.arange(0,64)[int(64*p6):64]
        C6_i= np.arange(0,64)[int(64*p6):64]
        C6_j = np.arange(0,64)[int(64*p6):64]
        C6_k = np.arange(0,64)[int(64*p6):64]

        #sorted(np.load(os.path.join(path,'conv_block3.conv2.weight.npy'))[int(256*p6):256])


        C7_r = np.arange(0,128)[int(128*p7):128]
        C7_i = np.arange(0,128)[int(128*p7):128]
        C7_j = np.arange(0,128)[int(128*p7):128]
        C7_k = np.arange(0,128)[int(128*p7):128]

        #sorted(np.load(os.path.join(path,'conv_block4.conv1.weight.npy'))[int(512*p7):512])
        C8_r = np.arange(0,128)[int(128*p8):128]
        C8_i = np.arange(0,128)[int(128*p8):128]
        C8_j = np.arange(0,128)[int(128*p8):128]
        C8_k = np.arange(0,128)[int(128*p8):128]
        #sorted(np.load(os.path.join(path,'conv_block4.conv2.weight.npy'))[int(512*p8):512])


        C9_r = np.arange(0,256)[int(256*p9):256]
        C9_i = np.arange(0,256)[int(256*p9):256]
        C9_j = np.arange(0,256)[int(256*p9):256]
        C9_k = np.arange(0,256)[int(256*p9):256]

        #sorted(np.load(os.path.join(path,'conv_block5.conv1.weight.npy'))[int(1024*p9):1024])
        C10_r = np.arange(0,256)[int(256*p10):256]
        C10_i = np.arange(0,256)[int(256*p10):256]
        C10_j = np.arange(0,256)[int(256*p10):256]
        C10_k = np.arange(0,256)[int(256*p10):256]
        ##sorted(np.load(os.path.join(path,'conv_block5.conv2.weight.npy'))[int(1024*p10):1024])

        C11_r = np.arange(0,512)[int(512*p11):512]
        C11_i = np.arange(0,512)[int(512*p11):512]
        C11_j = np.arange(0,512)[int(512*p11):512]
        C11_k = np.arange(0,512)[int(512*p11):512]

        #sorted(np.load(os.path.join(path,'conv_block6.conv1.weight.npy'))[int(2048*p11):2048])
        C12_r = np.arange(0,512)[int(512*p12):512]
        C12_i = np.arange(0,512)[int(512*p12):512]
        C12_j = np.arange(0,512)[int(512*p12):512]
        C12_k = np.arange(0,512)[int(512*p12):512]
        ##sorted(np.load(os.path.join(path,'conv_block6.conv2.weight.npy'))[int(2048*p12):2048])


        conv_index = OrderedDict()

        conv_index['conv_block1.conv1.r_weight'] = C1_r
        conv_index['conv_block1.conv1.i_weight'] = C1_i
        conv_index['conv_block1.conv1.j_weight'] = C1_j
        conv_index['conv_block1.conv1.k_weight'] = C1_k

        conv_index['conv_block1.conv2.r_weight'] = C2_r
        conv_index['conv_block1.conv2.i_weight'] = C2_i
        conv_index['conv_block1.conv2.j_weight'] = C2_j
        conv_index['conv_block1.conv2.k_weight'] = C2_k

        conv_index['conv_block2.conv1.r_weight'] = C3_r
        conv_index['conv_block2.conv1.i_weight'] = C3_i
        conv_index['conv_block2.conv1.j_weight'] = C3_j
        conv_index['conv_block2.conv1.k_weight'] = C3_k

        conv_index['conv_block2.conv2.r_weight'] = C4_r
        conv_index['conv_block2.conv2.i_weight'] = C4_i
        conv_index['conv_block2.conv2.j_weight'] = C4_j
        conv_index['conv_block2.conv2.k_weight'] = C4_k


        conv_index['conv_block3.conv1.r_weight'] = C5_r
        conv_index['conv_block3.conv1.i_weight'] = C5_i
        conv_index['conv_block3.conv1.j_weight'] = C5_j
        conv_index['conv_block3.conv1.k_weight'] = C5_k

        conv_index['conv_block3.conv2.r_weight'] = C6_r
        conv_index['conv_block3.conv2.i_weight'] = C6_i
        conv_index['conv_block3.conv2.j_weight'] = C6_j
        conv_index['conv_block3.conv2.k_weight'] = C6_k



        conv_index['conv_block4.conv1.r_weight'] = C7_r
        conv_index['conv_block4.conv1.i_weight'] = C7_i
        conv_index['conv_block4.conv1.j_weight'] = C7_j
        conv_index['conv_block4.conv1.k_weight'] = C7_k

        conv_index['conv_block4.conv2.r_weight'] = C8_r
        conv_index['conv_block4.conv2.i_weight'] = C8_i
        conv_index['conv_block4.conv2.j_weight'] = C8_j
        conv_index['conv_block4.conv2.k_weight'] = C8_k

        conv_index['conv_block5.conv1.r_weight'] = C9_r
        conv_index['conv_block5.conv1.i_weight'] = C9_i
        conv_index['conv_block5.conv1.j_weight'] = C9_j
        conv_index['conv_block5.conv1.k_weight'] = C9_k

        conv_index['conv_block5.conv2.r_weight'] = C10_r
        conv_index['conv_block5.conv2.i_weight'] = C10_i
        conv_index['conv_block5.conv2.j_weight'] = C10_j
        conv_index['conv_block5.conv2.k_weight'] = C10_k

        conv_index['conv_block6.conv1.r_weight'] = C11_r
        conv_index['conv_block6.conv1.i_weight'] = C11_i
        conv_index['conv_block6.conv1.j_weight'] = C11_j
        conv_index['conv_block6.conv1.k_weight'] = C11_k

        conv_index['conv_block6.conv2.r_weight'] = C12_r
        conv_index['conv_block6.conv2.i_weight'] = C12_i
        conv_index['conv_block6.conv2.j_weight'] = C12_j
        conv_index['conv_block6.conv2.k_weight'] = C12_k

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = QConvBlock_pruned(in_channels_1= 4,  out_channels_1=int(64*(1-p1)),out_channels_2=int(64*(1-p2)))
        self.conv_block2 = QConvBlock_pruned(in_channels_1=int(64*(1-p2)), out_channels_1=int(128*(1-p3)),out_channels_2=int(128*(1-p4)))
        self.conv_block3 = QConvBlock_pruned(in_channels_1=int(128*(1-p4)), out_channels_1=int(256*(1-p5)),out_channels_2=int(256*(1-p6)))
        self.conv_block4 = QConvBlock_pruned(in_channels_1=int(256*(1-p6)), out_channels_1=int(512*(1-p7)),out_channels_2=int(512*(1-p8)))
        self.conv_block5 = QConvBlock_pruned(in_channels_1=int(512*(1-p8)), out_channels_1=int(1024*(1-p9)),out_channels_2=int(1024*(1-p10)))
        self.conv_block6 = QConvBlock_pruned(in_channels_1=int(1024*(1-p10)), out_channels_1=int((1-p11)*2048),out_channels_2=int(2048*(1-p12)))

        self.fc1 = QuaternionLinear(int(2048*(1-p12)), 2048, bias=True) #nn.Linear(int(2048*(1-p12)), 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        # self.init_weight()
        checkpoint_path = '~/audio_tagging_quaternion-main/checkpoint/980000_iterations.pth'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        weights = checkpoint['model'] #['model']
        weights_pruned = weights #checkpoint['model'] #['model']


        # conv_key_list = ['conv_block1.conv1.weight', 'conv_block1.conv2.weight','conv_block2.conv1.weight', 'conv_block2.conv2.weight','conv_block3.conv1.weight', 'conv_block3.conv2.weight','conv_block4.conv1.weight', 'conv_block4.conv2.weight','conv_block5.conv1.weight', 'conv_block5.conv2.weight','conv_block6.conv1.weight', 'conv_block6.conv2.weight']
        bn_key_list = ['conv_block1.bn1.weight', 'conv_block1.bn1.bias', 'conv_block1.bn1.running_mean', 'conv_block1.bn1.running_var','conv_block1.bn2.weight', 'conv_block1.bn2.bias', 'conv_block1.bn2.running_mean', 'conv_block1.bn2.running_var','conv_block2.bn1.weight', 'conv_block2.bn1.bias', 'conv_block2.bn1.running_mean', 'conv_block2.bn1.running_var','conv_block2.bn2.weight', 'conv_block2.bn2.bias', 'conv_block2.bn2.running_mean', 'conv_block2.bn2.running_var','conv_block3.bn1.weight', 'conv_block3.bn1.bias', 'conv_block3.bn1.running_mean', 'conv_block3.bn1.running_var','conv_block3.bn2.weight', 'conv_block3.bn2.bias', 'conv_block3.bn2.running_mean', 'conv_block3.bn2.running_var','conv_block4.bn1.weight', 'conv_block4.bn1.bias', 'conv_block4.bn1.running_mean', 'conv_block4.bn1.running_var','conv_block4.bn2.weight', 'conv_block4.bn2.bias', 'conv_block4.bn2.running_mean', 'conv_block4.bn2.running_var','conv_block5.bn1.weight', 'conv_block5.bn1.bias', 'conv_block5.bn1.running_mean', 'conv_block5.bn1.running_var','conv_block5.bn2.weight', 'conv_block5.bn2.bias', 'conv_block5.bn2.running_mean', 'conv_block5.bn2.running_var','conv_block6.bn1.weight', 'conv_block6.bn1.bias', 'conv_block6.bn1.running_mean', 'conv_block6.bn1.running_var','conv_block6.bn2.weight', 'conv_block6.bn2.bias', 'conv_block6.bn2.running_mean', 'conv_block6.bn2.running_var']
        # prev_conv_key_list =  ['conv_block1.conv1.weight', 'conv_block1.conv2.weight','conv_block2.conv1.weight', 'conv_block2.conv2.weight','conv_block3.conv1.weight', 'conv_block3.conv2.weight','conv_block4.conv1.weight', 'conv_block4.conv2.weight','conv_block5.conv1.weight', 'conv_block5.conv2.weight','conv_block6.conv1.weight']


        conv_key_list = ['conv_block1.conv1.r_weight', 'conv_block1.conv1.i_weight', 'conv_block1.conv1.j_weight', 'conv_block1.conv1.k_weight',
                         'conv_block1.conv2.r_weight', 'conv_block1.conv2.i_weight', 'conv_block1.conv2.j_weight', 'conv_block1.conv2.k_weight',
                         'conv_block2.conv1.r_weight', 'conv_block2.conv1.i_weight', 'conv_block2.conv1.j_weight', 'conv_block2.conv1.k_weight',
                         'conv_block2.conv2.r_weight', 'conv_block2.conv2.i_weight', 'conv_block2.conv2.j_weight', 'conv_block2.conv2.k_weight',
                         'conv_block3.conv1.r_weight', 'conv_block3.conv1.i_weight', 'conv_block3.conv1.j_weight', 'conv_block3.conv1.k_weight',
                         'conv_block3.conv2.r_weight', 'conv_block3.conv2.i_weight', 'conv_block3.conv2.j_weight', 'conv_block3.conv2.k_weight',
                         'conv_block4.conv1.r_weight', 'conv_block4.conv1.i_weight', 'conv_block4.conv1.j_weight', 'conv_block4.conv1.k_weight',
                         'conv_block4.conv2.r_weight', 'conv_block4.conv2.i_weight', 'conv_block4.conv2.j_weight', 'conv_block4.conv2.k_weight',
                         'conv_block5.conv1.r_weight', 'conv_block5.conv1.i_weight', 'conv_block5.conv1.j_weight', 'conv_block5.conv1.k_weight',
                         'conv_block5.conv2.r_weight', 'conv_block5.conv2.i_weight', 'conv_block5.conv2.j_weight', 'conv_block5.conv2.k_weight',
                         'conv_block6.conv1.r_weight', 'conv_block6.conv1.i_weight', 'conv_block6.conv1.j_weight', 'conv_block6.conv1.k_weight',
                         'conv_block6.conv2.r_weight', 'conv_block6.conv2.i_weight', 'conv_block6.conv2.j_weight', 'conv_block6.conv2.k_weight']

        prev_conv_key_list = ['conv_block1.conv1.r_weight', 'conv_block1.conv1.i_weight', 'conv_block1.conv1.j_weight', 'conv_block1.conv1.k_weight',
                              'conv_block1.conv2.r_weight', 'conv_block1.conv2.i_weight', 'conv_block1.conv2.j_weight', 'conv_block1.conv2.k_weight',
                              'conv_block2.conv1.r_weight', 'conv_block2.conv1.i_weight', 'conv_block2.conv1.j_weight', 'conv_block2.conv1.k_weight',
                              'conv_block2.conv2.r_weight', 'conv_block2.conv2.i_weight', 'conv_block2.conv2.j_weight', 'conv_block2.conv2.k_weight',
                              'conv_block3.conv1.r_weight', 'conv_block3.conv1.i_weight', 'conv_block3.conv1.j_weight', 'conv_block3.conv1.k_weight',
                              'conv_block3.conv2.r_weight', 'conv_block3.conv2.i_weight', 'conv_block3.conv2.j_weight', 'conv_block3.conv2.k_weight',
                              'conv_block4.conv1.r_weight', 'conv_block4.conv1.i_weight', 'conv_block4.conv1.j_weight', 'conv_block4.conv1.k_weight',
                              'conv_block4.conv2.r_weight', 'conv_block4.conv2.i_weight', 'conv_block4.conv2.j_weight', 'conv_block4.conv2.k_weight',
                              'conv_block5.conv1.r_weight', 'conv_block5.conv1.i_weight', 'conv_block5.conv1.j_weight', 'conv_block5.conv1.k_weight',
                              'conv_block5.conv2.r_weight', 'conv_block5.conv2.i_weight', 'conv_block5.conv2.j_weight', 'conv_block5.conv2.k_weight',
                              'conv_block6.conv1.r_weight', 'conv_block6.conv1.i_weight', 'conv_block6.conv1.j_weight', 'conv_block6.conv1.k_weight',
                              'conv_block6.conv2.r_weight', 'conv_block6.conv2.i_weight', 'conv_block6.conv2.j_weight', 'conv_block6.conv2.k_weight']


        Z = OrderedDict()
        j = 0
        i = 0
        bn_i =0
        for key in conv_key_list:
            W_2D = weights[key].numpy()
            # print("i, key, W_2d,conv_index,weights_pruned {}".format(i), key, W_2D.shape,conv_index[key].shape ,weights_pruned[key].shape)
            if i <= 3:
                weights_pruned[key] = torch.tensor(W_2D[conv_index[key],:,:,:])
            else:
                weights_pruned[key] = torch.tensor(W_2D[conv_index[key],:,:,:][:,conv_index[prev_conv_key_list[i-4]],:,:])

            if i%4 == 0:
                # print(key,bn_key_list[j])
                # print(conv_key_list[i],conv_key_list[i+1],conv_key_list[i+2],conv_key_list[i+3]) 

                # print(weights[bn_key_list[j]].shape)
                # print(conv_index[key])
                # print(weights[bn_key_list[j]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+1]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+2]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+3]][conv_index[key]].shape)
                weights_pruned[bn_key_list[j]] = torch.cat((weights[bn_key_list[j]][conv_index[conv_key_list[i] ]], weights[bn_key_list[j]][conv_index[conv_key_list[i+1]]],
                                                            weights[bn_key_list[j]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j]][conv_index[conv_key_list[i+3]]]),dim=0) #weights[bn_key_list[j]][conv_index[key]]

                weights_pruned[bn_key_list[j+1]] =torch.cat((weights[bn_key_list[j+1]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+1]][conv_index[conv_key_list[i+1]]],
                                                             weights[bn_key_list[j+1]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+1]][conv_index[conv_key_list[i+3]]]), dim =0)

                weights_pruned[bn_key_list[j+2]] =torch.cat((weights[bn_key_list[j+2]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+2]][conv_index[conv_key_list[i+1]]],
                                                             weights[bn_key_list[j+2]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+2]][conv_index[conv_key_list[i+3]]]), dim=0)

                weights_pruned[bn_key_list[j+3]] = torch.cat((weights[bn_key_list[j+3]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+3]][conv_index[conv_key_list[i+1]]],
                                                              weights[bn_key_list[j+3]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+3]][conv_index[conv_key_list[i+3]]]), dim=0)

                j = j + 4



            i = i + 1
            # print(key)
            # weights[key] = torch.tensor(W_2D[0:50,:,:,:])
            filename = path + key + '.npy'
            # print("i, key, conv_index,weights_pruned {}".format(i), key, W_2D.shape,conv_index[key].shape ,weights_pruned[key].shape)
            # print(filename)
            # np.save(filename,sorted_index)
            # print(len(sorted_index))


        weights_pruned['fc1.r_weight'] = weights['fc1.r_weight'][conv_index['conv_block6.conv2.r_weight'],:]
        weights_pruned['fc1.i_weight'] = weights['fc1.i_weight'][conv_index['conv_block6.conv2.i_weight'],:]
        weights_pruned['fc1.j_weight'] = weights['fc1.j_weight'][conv_index['conv_block6.conv2.j_weight'],:]
        weights_pruned['fc1.k_weight'] = weights['fc1.k_weight'][conv_index['conv_block6.conv2.k_weight'],:]

        self.load_state_dict(weights_pruned)


    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)


        x_first = torchaudio.functional.compute_deltas(x)
        x_second = torchaudio.functional.compute_deltas(x_first)
        x_third = torchaudio.functional.compute_deltas(x_second)
        #quternionic converter
        x_quaternion = torch.cat([x,x_first, x_second, x_third], dim=1) # (batch_size, 4, time_steps, freq_bins)
        #print("spectrogram shape", x_quaternion.shape)
        x = self.logmel_extractor(x_quaternion)
        # x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        # clipwise_output = torch.log_softmax(self.fc_audioset(x))        
        # clipwise_output = nn.functional.softmax(self.fc_audioset(x))        

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
    
    
class QCnn14_pruned_50(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
            fmax, classes_num):#, pooling_type, pooling_factor):
        import os
        import numpy as np

        super(QCnn14_pruned_50, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        from collections import OrderedDict
        path = '~/audio_tagging_quaternion-main/sorted_index_L1_pruning/' #'/home/arshdeep/Pruning/PANNs_pruning/sorted_Indexes/OP/'   

        p = 0.50
        p1 = 0
        p2 = 0
        p3 = 0
        p4 = 0
        p5 = 0
        p6 = 0
        p7 = p
        p8 = p
        p9 = p
        p10 = p
        p11 = p
        p12 = p

        C1_r = np.arange(0,16)[int(16*p1):16]#sorted(np.load(os.path.join(path,'conv_block1.conv1.weight.npy'))[int(64*p1):64])
        C1_i = np.arange(0,16)[int(16*p1):16]
        C1_j = np.arange(0,16)[int(16*p1):16]
        C1_k = np.arange(0,16)[int(16*p1):16]
        # C1_k = np.arange(0,100)[16+int(16*p1):32]
        # 
        C2_r= np.arange(0,16)[int(16*p2):16]#sorted(np.load(os.path.join(path,'conv_block1.conv2.weight.npy'))[int(64*p2):64])
        C2_i= np.arange(0,16)[int(16*p2):16]
        C2_j= np.arange(0,16)[int(16*p2):16]
        C2_k= np.arange(0,16)[int(16*p2):16]

        C3_r = np.arange(0,32)[int(32*p3):32]
        C3_i = np.arange(0,32)[int(32*p3):32]
        C3_j = np.arange(0,32)[int(32*p3):32]
        C3_k = np.arange(0,32)[int(32*p3):32]


        #sorted(np.load(os.path.join(path,'conv_block2.conv1.weight.npy'))[int(128*p3):128])
        C4_r = np.arange(0,32)[int(32*p4):32]
        C4_i = np.arange(0,32)[int(32*p4):32]
        C4_j = np.arange(0,32)[int(32*p4):32]
        C4_k = np.arange(0,32)[int(32*p4):32]


        #sorted(np.load(os.path.join(path,'conv_block2.conv2.weight.npy'))[int(128*p4):128])


        C5_r = np.arange(0,64)[int(64*p5):64]
        C5_i = np.arange(0,64)[int(64*p5):64]
        C5_j = np.arange(0,64)[int(64*p5):64]
        C5_k = np.arange(0,64)[int(64*p5):64]

        #sorted(np.load(os.path.join(path,'conv_block3.conv1.weight.npy'))[int(256*p5):256])
        C6_r = np.arange(0,64)[int(64*p6):64]
        C6_i= np.arange(0,64)[int(64*p6):64]
        C6_j = np.arange(0,64)[int(64*p6):64]
        C6_k = np.arange(0,64)[int(64*p6):64]

        #sorted(np.load(os.path.join(path,'conv_block3.conv2.weight.npy'))[int(256*p6):256])


        C7_r = np.arange(0,128)[int(128*p7):128]
        C7_i = np.arange(0,128)[int(128*p7):128]
        C7_j = np.arange(0,128)[int(128*p7):128]
        C7_k = np.arange(0,128)[int(128*p7):128]

        #sorted(np.load(os.path.join(path,'conv_block4.conv1.weight.npy'))[int(512*p7):512])
        C8_r = np.arange(0,128)[int(128*p8):128]
        C8_i = np.arange(0,128)[int(128*p8):128]
        C8_j = np.arange(0,128)[int(128*p8):128]
        C8_k = np.arange(0,128)[int(128*p8):128]
        #sorted(np.load(os.path.join(path,'conv_block4.conv2.weight.npy'))[int(512*p8):512])


        C9_r = np.arange(0,256)[int(256*p9):256]
        C9_i = np.arange(0,256)[int(256*p9):256]
        C9_j = np.arange(0,256)[int(256*p9):256]
        C9_k = np.arange(0,256)[int(256*p9):256]

        #sorted(np.load(os.path.join(path,'conv_block5.conv1.weight.npy'))[int(1024*p9):1024])
        C10_r = np.arange(0,256)[int(256*p10):256]
        C10_i = np.arange(0,256)[int(256*p10):256]
        C10_j = np.arange(0,256)[int(256*p10):256]
        C10_k = np.arange(0,256)[int(256*p10):256]
        ##sorted(np.load(os.path.join(path,'conv_block5.conv2.weight.npy'))[int(1024*p10):1024])

        C11_r = np.arange(0,512)[int(512*p11):512]
        C11_i = np.arange(0,512)[int(512*p11):512]
        C11_j = np.arange(0,512)[int(512*p11):512]
        C11_k = np.arange(0,512)[int(512*p11):512]

        #sorted(np.load(os.path.join(path,'conv_block6.conv1.weight.npy'))[int(2048*p11):2048])
        C12_r = np.arange(0,512)[int(512*p12):512]
        C12_i = np.arange(0,512)[int(512*p12):512]
        C12_j = np.arange(0,512)[int(512*p12):512]
        C12_k = np.arange(0,512)[int(512*p12):512]
        ##sorted(np.load(os.path.join(path,'conv_block6.conv2.weight.npy'))[int(2048*p12):2048])


        conv_index = OrderedDict()

        conv_index['conv_block1.conv1.r_weight'] = C1_r
        conv_index['conv_block1.conv1.i_weight'] = C1_i
        conv_index['conv_block1.conv1.j_weight'] = C1_j
        conv_index['conv_block1.conv1.k_weight'] = C1_k

        conv_index['conv_block1.conv2.r_weight'] = C2_r
        conv_index['conv_block1.conv2.i_weight'] = C2_i
        conv_index['conv_block1.conv2.j_weight'] = C2_j
        conv_index['conv_block1.conv2.k_weight'] = C2_k

        conv_index['conv_block2.conv1.r_weight'] = C3_r
        conv_index['conv_block2.conv1.i_weight'] = C3_i
        conv_index['conv_block2.conv1.j_weight'] = C3_j
        conv_index['conv_block2.conv1.k_weight'] = C3_k

        conv_index['conv_block2.conv2.r_weight'] = C4_r
        conv_index['conv_block2.conv2.i_weight'] = C4_i
        conv_index['conv_block2.conv2.j_weight'] = C4_j
        conv_index['conv_block2.conv2.k_weight'] = C4_k


        conv_index['conv_block3.conv1.r_weight'] = C5_r
        conv_index['conv_block3.conv1.i_weight'] = C5_i
        conv_index['conv_block3.conv1.j_weight'] = C5_j
        conv_index['conv_block3.conv1.k_weight'] = C5_k

        conv_index['conv_block3.conv2.r_weight'] = C6_r
        conv_index['conv_block3.conv2.i_weight'] = C6_i
        conv_index['conv_block3.conv2.j_weight'] = C6_j
        conv_index['conv_block3.conv2.k_weight'] = C6_k



        conv_index['conv_block4.conv1.r_weight'] = C7_r
        conv_index['conv_block4.conv1.i_weight'] = C7_i
        conv_index['conv_block4.conv1.j_weight'] = C7_j
        conv_index['conv_block4.conv1.k_weight'] = C7_k

        conv_index['conv_block4.conv2.r_weight'] = C8_r
        conv_index['conv_block4.conv2.i_weight'] = C8_i
        conv_index['conv_block4.conv2.j_weight'] = C8_j
        conv_index['conv_block4.conv2.k_weight'] = C8_k

        conv_index['conv_block5.conv1.r_weight'] = C9_r
        conv_index['conv_block5.conv1.i_weight'] = C9_i
        conv_index['conv_block5.conv1.j_weight'] = C9_j
        conv_index['conv_block5.conv1.k_weight'] = C9_k

        conv_index['conv_block5.conv2.r_weight'] = C10_r
        conv_index['conv_block5.conv2.i_weight'] = C10_i
        conv_index['conv_block5.conv2.j_weight'] = C10_j
        conv_index['conv_block5.conv2.k_weight'] = C10_k

        conv_index['conv_block6.conv1.r_weight'] = C11_r
        conv_index['conv_block6.conv1.i_weight'] = C11_i
        conv_index['conv_block6.conv1.j_weight'] = C11_j
        conv_index['conv_block6.conv1.k_weight'] = C11_k

        conv_index['conv_block6.conv2.r_weight'] = C12_r
        conv_index['conv_block6.conv2.i_weight'] = C12_i
        conv_index['conv_block6.conv2.j_weight'] = C12_j
        conv_index['conv_block6.conv2.k_weight'] = C12_k

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = QConvBlock_pruned(in_channels_1= 4,  out_channels_1=int(64*(1-p1)),out_channels_2=int(64*(1-p2)))
        self.conv_block2 = QConvBlock_pruned(in_channels_1=int(64*(1-p2)), out_channels_1=int(128*(1-p3)),out_channels_2=int(128*(1-p4)))
        self.conv_block3 = QConvBlock_pruned(in_channels_1=int(128*(1-p4)), out_channels_1=int(256*(1-p5)),out_channels_2=int(256*(1-p6)))
        self.conv_block4 = QConvBlock_pruned(in_channels_1=int(256*(1-p6)), out_channels_1=int(512*(1-p7)),out_channels_2=int(512*(1-p8)))
        self.conv_block5 = QConvBlock_pruned(in_channels_1=int(512*(1-p8)), out_channels_1=int(1024*(1-p9)),out_channels_2=int(1024*(1-p10)))
        self.conv_block6 = QConvBlock_pruned(in_channels_1=int(1024*(1-p10)), out_channels_1=int((1-p11)*2048),out_channels_2=int(2048*(1-p12)))

        self.fc1 = QuaternionLinear(int(2048*(1-p12)), 2048, bias=True) #nn.Linear(int(2048*(1-p12)), 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        # self.init_weight()
        checkpoint_path = '~/audio_tagging_quaternion-main/checkpoint/980000_iterations.pth'#
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        weights = checkpoint['model'] #['model']
        weights_pruned = weights #checkpoint['model'] #['model']


        # conv_key_list = ['conv_block1.conv1.weight', 'conv_block1.conv2.weight','conv_block2.conv1.weight', 'conv_block2.conv2.weight','conv_block3.conv1.weight', 'conv_block3.conv2.weight','conv_block4.conv1.weight', 'conv_block4.conv2.weight','conv_block5.conv1.weight', 'conv_block5.conv2.weight','conv_block6.conv1.weight', 'conv_block6.conv2.weight']
        bn_key_list = ['conv_block1.bn1.weight', 'conv_block1.bn1.bias', 'conv_block1.bn1.running_mean', 'conv_block1.bn1.running_var','conv_block1.bn2.weight', 'conv_block1.bn2.bias', 'conv_block1.bn2.running_mean', 'conv_block1.bn2.running_var','conv_block2.bn1.weight', 'conv_block2.bn1.bias', 'conv_block2.bn1.running_mean', 'conv_block2.bn1.running_var','conv_block2.bn2.weight', 'conv_block2.bn2.bias', 'conv_block2.bn2.running_mean', 'conv_block2.bn2.running_var','conv_block3.bn1.weight', 'conv_block3.bn1.bias', 'conv_block3.bn1.running_mean', 'conv_block3.bn1.running_var','conv_block3.bn2.weight', 'conv_block3.bn2.bias', 'conv_block3.bn2.running_mean', 'conv_block3.bn2.running_var','conv_block4.bn1.weight', 'conv_block4.bn1.bias', 'conv_block4.bn1.running_mean', 'conv_block4.bn1.running_var','conv_block4.bn2.weight', 'conv_block4.bn2.bias', 'conv_block4.bn2.running_mean', 'conv_block4.bn2.running_var','conv_block5.bn1.weight', 'conv_block5.bn1.bias', 'conv_block5.bn1.running_mean', 'conv_block5.bn1.running_var','conv_block5.bn2.weight', 'conv_block5.bn2.bias', 'conv_block5.bn2.running_mean', 'conv_block5.bn2.running_var','conv_block6.bn1.weight', 'conv_block6.bn1.bias', 'conv_block6.bn1.running_mean', 'conv_block6.bn1.running_var','conv_block6.bn2.weight', 'conv_block6.bn2.bias', 'conv_block6.bn2.running_mean', 'conv_block6.bn2.running_var']
        # prev_conv_key_list =  ['conv_block1.conv1.weight', 'conv_block1.conv2.weight','conv_block2.conv1.weight', 'conv_block2.conv2.weight','conv_block3.conv1.weight', 'conv_block3.conv2.weight','conv_block4.conv1.weight', 'conv_block4.conv2.weight','conv_block5.conv1.weight', 'conv_block5.conv2.weight','conv_block6.conv1.weight']


        conv_key_list = ['conv_block1.conv1.r_weight', 'conv_block1.conv1.i_weight', 'conv_block1.conv1.j_weight', 'conv_block1.conv1.k_weight',
                         'conv_block1.conv2.r_weight', 'conv_block1.conv2.i_weight', 'conv_block1.conv2.j_weight', 'conv_block1.conv2.k_weight',
                         'conv_block2.conv1.r_weight', 'conv_block2.conv1.i_weight', 'conv_block2.conv1.j_weight', 'conv_block2.conv1.k_weight',
                         'conv_block2.conv2.r_weight', 'conv_block2.conv2.i_weight', 'conv_block2.conv2.j_weight', 'conv_block2.conv2.k_weight',
                         'conv_block3.conv1.r_weight', 'conv_block3.conv1.i_weight', 'conv_block3.conv1.j_weight', 'conv_block3.conv1.k_weight',
                         'conv_block3.conv2.r_weight', 'conv_block3.conv2.i_weight', 'conv_block3.conv2.j_weight', 'conv_block3.conv2.k_weight',
                         'conv_block4.conv1.r_weight', 'conv_block4.conv1.i_weight', 'conv_block4.conv1.j_weight', 'conv_block4.conv1.k_weight',
                         'conv_block4.conv2.r_weight', 'conv_block4.conv2.i_weight', 'conv_block4.conv2.j_weight', 'conv_block4.conv2.k_weight',
                         'conv_block5.conv1.r_weight', 'conv_block5.conv1.i_weight', 'conv_block5.conv1.j_weight', 'conv_block5.conv1.k_weight',
                         'conv_block5.conv2.r_weight', 'conv_block5.conv2.i_weight', 'conv_block5.conv2.j_weight', 'conv_block5.conv2.k_weight',
                         'conv_block6.conv1.r_weight', 'conv_block6.conv1.i_weight', 'conv_block6.conv1.j_weight', 'conv_block6.conv1.k_weight',
                         'conv_block6.conv2.r_weight', 'conv_block6.conv2.i_weight', 'conv_block6.conv2.j_weight', 'conv_block6.conv2.k_weight']

        prev_conv_key_list = ['conv_block1.conv1.r_weight', 'conv_block1.conv1.i_weight', 'conv_block1.conv1.j_weight', 'conv_block1.conv1.k_weight',
                              'conv_block1.conv2.r_weight', 'conv_block1.conv2.i_weight', 'conv_block1.conv2.j_weight', 'conv_block1.conv2.k_weight',
                              'conv_block2.conv1.r_weight', 'conv_block2.conv1.i_weight', 'conv_block2.conv1.j_weight', 'conv_block2.conv1.k_weight',
                              'conv_block2.conv2.r_weight', 'conv_block2.conv2.i_weight', 'conv_block2.conv2.j_weight', 'conv_block2.conv2.k_weight',
                              'conv_block3.conv1.r_weight', 'conv_block3.conv1.i_weight', 'conv_block3.conv1.j_weight', 'conv_block3.conv1.k_weight',
                              'conv_block3.conv2.r_weight', 'conv_block3.conv2.i_weight', 'conv_block3.conv2.j_weight', 'conv_block3.conv2.k_weight',
                              'conv_block4.conv1.r_weight', 'conv_block4.conv1.i_weight', 'conv_block4.conv1.j_weight', 'conv_block4.conv1.k_weight',
                              'conv_block4.conv2.r_weight', 'conv_block4.conv2.i_weight', 'conv_block4.conv2.j_weight', 'conv_block4.conv2.k_weight',
                              'conv_block5.conv1.r_weight', 'conv_block5.conv1.i_weight', 'conv_block5.conv1.j_weight', 'conv_block5.conv1.k_weight',
                              'conv_block5.conv2.r_weight', 'conv_block5.conv2.i_weight', 'conv_block5.conv2.j_weight', 'conv_block5.conv2.k_weight',
                              'conv_block6.conv1.r_weight', 'conv_block6.conv1.i_weight', 'conv_block6.conv1.j_weight', 'conv_block6.conv1.k_weight',
                              'conv_block6.conv2.r_weight', 'conv_block6.conv2.i_weight', 'conv_block6.conv2.j_weight', 'conv_block6.conv2.k_weight']


        Z = OrderedDict()
        j = 0
        i = 0
        bn_i =0
        for key in conv_key_list:
            W_2D = weights[key].numpy()
            # print("i, key, W_2d,conv_index,weights_pruned {}".format(i), key, W_2D.shape,conv_index[key].shape ,weights_pruned[key].shape)
            if i <= 3:
                weights_pruned[key] = torch.tensor(W_2D[conv_index[key],:,:,:])
            else:
                weights_pruned[key] = torch.tensor(W_2D[conv_index[key],:,:,:][:,conv_index[prev_conv_key_list[i-4]],:,:])

            if i%4 == 0:
                # print(key,bn_key_list[j])
                # print(conv_key_list[i],conv_key_list[i+1],conv_key_list[i+2],conv_key_list[i+3]) 

                # print(weights[bn_key_list[j]].shape)
                # print(conv_index[key])
                # print(weights[bn_key_list[j]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+1]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+2]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+3]][conv_index[key]].shape)
                weights_pruned[bn_key_list[j]] = torch.cat((weights[bn_key_list[j]][conv_index[conv_key_list[i] ]], weights[bn_key_list[j]][conv_index[conv_key_list[i+1]]],
                                                            weights[bn_key_list[j]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j]][conv_index[conv_key_list[i+3]]]),dim=0) #weights[bn_key_list[j]][conv_index[key]]

                weights_pruned[bn_key_list[j+1]] =torch.cat((weights[bn_key_list[j+1]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+1]][conv_index[conv_key_list[i+1]]],
                                                             weights[bn_key_list[j+1]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+1]][conv_index[conv_key_list[i+3]]]), dim =0)

                weights_pruned[bn_key_list[j+2]] =torch.cat((weights[bn_key_list[j+2]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+2]][conv_index[conv_key_list[i+1]]],
                                                             weights[bn_key_list[j+2]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+2]][conv_index[conv_key_list[i+3]]]), dim=0)

                weights_pruned[bn_key_list[j+3]] = torch.cat((weights[bn_key_list[j+3]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+3]][conv_index[conv_key_list[i+1]]],
                                                              weights[bn_key_list[j+3]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+3]][conv_index[conv_key_list[i+3]]]), dim=0)

                j = j + 4



            i = i + 1
            # print(key)
            # weights[key] = torch.tensor(W_2D[0:50,:,:,:])
            filename = path + key + '.npy'
            # print("i, key, conv_index,weights_pruned {}".format(i), key, W_2D.shape,conv_index[key].shape ,weights_pruned[key].shape)
            # print(filename)
            # np.save(filename,sorted_index)
            # print(len(sorted_index))


        weights_pruned['fc1.r_weight'] = weights['fc1.r_weight'][conv_index['conv_block6.conv2.r_weight'],:]
        weights_pruned['fc1.i_weight'] = weights['fc1.i_weight'][conv_index['conv_block6.conv2.i_weight'],:]
        weights_pruned['fc1.j_weight'] = weights['fc1.j_weight'][conv_index['conv_block6.conv2.j_weight'],:]
        weights_pruned['fc1.k_weight'] = weights['fc1.k_weight'][conv_index['conv_block6.conv2.k_weight'],:]

        self.load_state_dict(weights_pruned)


    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)


        x_first = torchaudio.functional.compute_deltas(x)
        x_second = torchaudio.functional.compute_deltas(x_first)
        x_third = torchaudio.functional.compute_deltas(x_second)
        #quternionic converter
        x_quaternion = torch.cat([x,x_first, x_second, x_third], dim=1) # (batch_size, 4, time_steps, freq_bins)
        #print("spectrogram shape", x_quaternion.shape)
        x = self.logmel_extractor(x_quaternion)
        # x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        # clipwise_output = torch.log_softmax(self.fc_audioset(x))        
        # clipwise_output = nn.functional.softmax(self.fc_audioset(x))        

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict    
    
    
class QCnn14_pruned_75(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
            fmax, classes_num):#, pooling_type, pooling_factor):
        import os
        import numpy as np

        super(QCnn14_pruned_75, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        from collections import OrderedDict
        path = '~/audio_tagging_quaternion-main/sorted_index_L1_pruning/' #'/home/arshdeep/Pruning/PANNs_pruning/sorted_Indexes/OP/'   

        p = 0.75
        p1 = 0
        p2 = 0
        p3 = 0
        p4 = 0
        p5 = 0
        p6 = 0
        p7 = p
        p8 = p
        p9 = p
        p10 = p
        p11 = p
        p12 = p

        C1_r = np.arange(0,16)[int(16*p1):16]#sorted(np.load(os.path.join(path,'conv_block1.conv1.weight.npy'))[int(64*p1):64])
        C1_i = np.arange(0,16)[int(16*p1):16]
        C1_j = np.arange(0,16)[int(16*p1):16]
        C1_k = np.arange(0,16)[int(16*p1):16]
        # C1_k = np.arange(0,100)[16+int(16*p1):32]
        # 
        C2_r= np.arange(0,16)[int(16*p2):16]#sorted(np.load(os.path.join(path,'conv_block1.conv2.weight.npy'))[int(64*p2):64])
        C2_i= np.arange(0,16)[int(16*p2):16]
        C2_j= np.arange(0,16)[int(16*p2):16]
        C2_k= np.arange(0,16)[int(16*p2):16]

        C3_r = np.arange(0,32)[int(32*p3):32]
        C3_i = np.arange(0,32)[int(32*p3):32]
        C3_j = np.arange(0,32)[int(32*p3):32]
        C3_k = np.arange(0,32)[int(32*p3):32]


        #sorted(np.load(os.path.join(path,'conv_block2.conv1.weight.npy'))[int(128*p3):128])
        C4_r = np.arange(0,32)[int(32*p4):32]
        C4_i = np.arange(0,32)[int(32*p4):32]
        C4_j = np.arange(0,32)[int(32*p4):32]
        C4_k = np.arange(0,32)[int(32*p4):32]


        #sorted(np.load(os.path.join(path,'conv_block2.conv2.weight.npy'))[int(128*p4):128])


        C5_r = np.arange(0,64)[int(64*p5):64]
        C5_i = np.arange(0,64)[int(64*p5):64]
        C5_j = np.arange(0,64)[int(64*p5):64]
        C5_k = np.arange(0,64)[int(64*p5):64]

        #sorted(np.load(os.path.join(path,'conv_block3.conv1.weight.npy'))[int(256*p5):256])
        C6_r = np.arange(0,64)[int(64*p6):64]
        C6_i= np.arange(0,64)[int(64*p6):64]
        C6_j = np.arange(0,64)[int(64*p6):64]
        C6_k = np.arange(0,64)[int(64*p6):64]

        #sorted(np.load(os.path.join(path,'conv_block3.conv2.weight.npy'))[int(256*p6):256])


        C7_r = np.arange(0,128)[int(128*p7):128]
        C7_i = np.arange(0,128)[int(128*p7):128]
        C7_j = np.arange(0,128)[int(128*p7):128]
        C7_k = np.arange(0,128)[int(128*p7):128]

        #sorted(np.load(os.path.join(path,'conv_block4.conv1.weight.npy'))[int(512*p7):512])
        C8_r = np.arange(0,128)[int(128*p8):128]
        C8_i = np.arange(0,128)[int(128*p8):128]
        C8_j = np.arange(0,128)[int(128*p8):128]
        C8_k = np.arange(0,128)[int(128*p8):128]
        #sorted(np.load(os.path.join(path,'conv_block4.conv2.weight.npy'))[int(512*p8):512])


        C9_r = np.arange(0,256)[int(256*p9):256]
        C9_i = np.arange(0,256)[int(256*p9):256]
        C9_j = np.arange(0,256)[int(256*p9):256]
        C9_k = np.arange(0,256)[int(256*p9):256]

        #sorted(np.load(os.path.join(path,'conv_block5.conv1.weight.npy'))[int(1024*p9):1024])
        C10_r = np.arange(0,256)[int(256*p10):256]
        C10_i = np.arange(0,256)[int(256*p10):256]
        C10_j = np.arange(0,256)[int(256*p10):256]
        C10_k = np.arange(0,256)[int(256*p10):256]
        ##sorted(np.load(os.path.join(path,'conv_block5.conv2.weight.npy'))[int(1024*p10):1024])

        C11_r = np.arange(0,512)[int(512*p11):512]
        C11_i = np.arange(0,512)[int(512*p11):512]
        C11_j = np.arange(0,512)[int(512*p11):512]
        C11_k = np.arange(0,512)[int(512*p11):512]

        #sorted(np.load(os.path.join(path,'conv_block6.conv1.weight.npy'))[int(2048*p11):2048])
        C12_r = np.arange(0,512)[int(512*p12):512]
        C12_i = np.arange(0,512)[int(512*p12):512]
        C12_j = np.arange(0,512)[int(512*p12):512]
        C12_k = np.arange(0,512)[int(512*p12):512]
        ##sorted(np.load(os.path.join(path,'conv_block6.conv2.weight.npy'))[int(2048*p12):2048])


        conv_index = OrderedDict()

        conv_index['conv_block1.conv1.r_weight'] = C1_r
        conv_index['conv_block1.conv1.i_weight'] = C1_i
        conv_index['conv_block1.conv1.j_weight'] = C1_j
        conv_index['conv_block1.conv1.k_weight'] = C1_k

        conv_index['conv_block1.conv2.r_weight'] = C2_r
        conv_index['conv_block1.conv2.i_weight'] = C2_i
        conv_index['conv_block1.conv2.j_weight'] = C2_j
        conv_index['conv_block1.conv2.k_weight'] = C2_k

        conv_index['conv_block2.conv1.r_weight'] = C3_r
        conv_index['conv_block2.conv1.i_weight'] = C3_i
        conv_index['conv_block2.conv1.j_weight'] = C3_j
        conv_index['conv_block2.conv1.k_weight'] = C3_k

        conv_index['conv_block2.conv2.r_weight'] = C4_r
        conv_index['conv_block2.conv2.i_weight'] = C4_i
        conv_index['conv_block2.conv2.j_weight'] = C4_j
        conv_index['conv_block2.conv2.k_weight'] = C4_k


        conv_index['conv_block3.conv1.r_weight'] = C5_r
        conv_index['conv_block3.conv1.i_weight'] = C5_i
        conv_index['conv_block3.conv1.j_weight'] = C5_j
        conv_index['conv_block3.conv1.k_weight'] = C5_k

        conv_index['conv_block3.conv2.r_weight'] = C6_r
        conv_index['conv_block3.conv2.i_weight'] = C6_i
        conv_index['conv_block3.conv2.j_weight'] = C6_j
        conv_index['conv_block3.conv2.k_weight'] = C6_k



        conv_index['conv_block4.conv1.r_weight'] = C7_r
        conv_index['conv_block4.conv1.i_weight'] = C7_i
        conv_index['conv_block4.conv1.j_weight'] = C7_j
        conv_index['conv_block4.conv1.k_weight'] = C7_k

        conv_index['conv_block4.conv2.r_weight'] = C8_r
        conv_index['conv_block4.conv2.i_weight'] = C8_i
        conv_index['conv_block4.conv2.j_weight'] = C8_j
        conv_index['conv_block4.conv2.k_weight'] = C8_k

        conv_index['conv_block5.conv1.r_weight'] = C9_r
        conv_index['conv_block5.conv1.i_weight'] = C9_i
        conv_index['conv_block5.conv1.j_weight'] = C9_j
        conv_index['conv_block5.conv1.k_weight'] = C9_k

        conv_index['conv_block5.conv2.r_weight'] = C10_r
        conv_index['conv_block5.conv2.i_weight'] = C10_i
        conv_index['conv_block5.conv2.j_weight'] = C10_j
        conv_index['conv_block5.conv2.k_weight'] = C10_k

        conv_index['conv_block6.conv1.r_weight'] = C11_r
        conv_index['conv_block6.conv1.i_weight'] = C11_i
        conv_index['conv_block6.conv1.j_weight'] = C11_j
        conv_index['conv_block6.conv1.k_weight'] = C11_k

        conv_index['conv_block6.conv2.r_weight'] = C12_r
        conv_index['conv_block6.conv2.i_weight'] = C12_i
        conv_index['conv_block6.conv2.j_weight'] = C12_j
        conv_index['conv_block6.conv2.k_weight'] = C12_k

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = QConvBlock_pruned(in_channels_1= 4,  out_channels_1=int(64*(1-p1)),out_channels_2=int(64*(1-p2)))
        self.conv_block2 = QConvBlock_pruned(in_channels_1=int(64*(1-p2)), out_channels_1=int(128*(1-p3)),out_channels_2=int(128*(1-p4)))
        self.conv_block3 = QConvBlock_pruned(in_channels_1=int(128*(1-p4)), out_channels_1=int(256*(1-p5)),out_channels_2=int(256*(1-p6)))
        self.conv_block4 = QConvBlock_pruned(in_channels_1=int(256*(1-p6)), out_channels_1=int(512*(1-p7)),out_channels_2=int(512*(1-p8)))
        self.conv_block5 = QConvBlock_pruned(in_channels_1=int(512*(1-p8)), out_channels_1=int(1024*(1-p9)),out_channels_2=int(1024*(1-p10)))
        self.conv_block6 = QConvBlock_pruned(in_channels_1=int(1024*(1-p10)), out_channels_1=int((1-p11)*2048),out_channels_2=int(2048*(1-p12)))

        self.fc1 = QuaternionLinear(int(2048*(1-p12)), 2048, bias=True) #nn.Linear(int(2048*(1-p12)), 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        # self.init_weight()
        checkpoint_path = '~/audio_tagging_quaternion-main/checkpoint/980000_iterations.pth'#
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        weights = checkpoint['model'] #['model']
        weights_pruned = weights #checkpoint['model'] #['model']


        # conv_key_list = ['conv_block1.conv1.weight', 'conv_block1.conv2.weight','conv_block2.conv1.weight', 'conv_block2.conv2.weight','conv_block3.conv1.weight', 'conv_block3.conv2.weight','conv_block4.conv1.weight', 'conv_block4.conv2.weight','conv_block5.conv1.weight', 'conv_block5.conv2.weight','conv_block6.conv1.weight', 'conv_block6.conv2.weight']
        bn_key_list = ['conv_block1.bn1.weight', 'conv_block1.bn1.bias', 'conv_block1.bn1.running_mean', 'conv_block1.bn1.running_var','conv_block1.bn2.weight', 'conv_block1.bn2.bias', 'conv_block1.bn2.running_mean', 'conv_block1.bn2.running_var','conv_block2.bn1.weight', 'conv_block2.bn1.bias', 'conv_block2.bn1.running_mean', 'conv_block2.bn1.running_var','conv_block2.bn2.weight', 'conv_block2.bn2.bias', 'conv_block2.bn2.running_mean', 'conv_block2.bn2.running_var','conv_block3.bn1.weight', 'conv_block3.bn1.bias', 'conv_block3.bn1.running_mean', 'conv_block3.bn1.running_var','conv_block3.bn2.weight', 'conv_block3.bn2.bias', 'conv_block3.bn2.running_mean', 'conv_block3.bn2.running_var','conv_block4.bn1.weight', 'conv_block4.bn1.bias', 'conv_block4.bn1.running_mean', 'conv_block4.bn1.running_var','conv_block4.bn2.weight', 'conv_block4.bn2.bias', 'conv_block4.bn2.running_mean', 'conv_block4.bn2.running_var','conv_block5.bn1.weight', 'conv_block5.bn1.bias', 'conv_block5.bn1.running_mean', 'conv_block5.bn1.running_var','conv_block5.bn2.weight', 'conv_block5.bn2.bias', 'conv_block5.bn2.running_mean', 'conv_block5.bn2.running_var','conv_block6.bn1.weight', 'conv_block6.bn1.bias', 'conv_block6.bn1.running_mean', 'conv_block6.bn1.running_var','conv_block6.bn2.weight', 'conv_block6.bn2.bias', 'conv_block6.bn2.running_mean', 'conv_block6.bn2.running_var']
        # prev_conv_key_list =  ['conv_block1.conv1.weight', 'conv_block1.conv2.weight','conv_block2.conv1.weight', 'conv_block2.conv2.weight','conv_block3.conv1.weight', 'conv_block3.conv2.weight','conv_block4.conv1.weight', 'conv_block4.conv2.weight','conv_block5.conv1.weight', 'conv_block5.conv2.weight','conv_block6.conv1.weight']


        conv_key_list = ['conv_block1.conv1.r_weight', 'conv_block1.conv1.i_weight', 'conv_block1.conv1.j_weight', 'conv_block1.conv1.k_weight',
                         'conv_block1.conv2.r_weight', 'conv_block1.conv2.i_weight', 'conv_block1.conv2.j_weight', 'conv_block1.conv2.k_weight',
                         'conv_block2.conv1.r_weight', 'conv_block2.conv1.i_weight', 'conv_block2.conv1.j_weight', 'conv_block2.conv1.k_weight',
                         'conv_block2.conv2.r_weight', 'conv_block2.conv2.i_weight', 'conv_block2.conv2.j_weight', 'conv_block2.conv2.k_weight',
                         'conv_block3.conv1.r_weight', 'conv_block3.conv1.i_weight', 'conv_block3.conv1.j_weight', 'conv_block3.conv1.k_weight',
                         'conv_block3.conv2.r_weight', 'conv_block3.conv2.i_weight', 'conv_block3.conv2.j_weight', 'conv_block3.conv2.k_weight',
                         'conv_block4.conv1.r_weight', 'conv_block4.conv1.i_weight', 'conv_block4.conv1.j_weight', 'conv_block4.conv1.k_weight',
                         'conv_block4.conv2.r_weight', 'conv_block4.conv2.i_weight', 'conv_block4.conv2.j_weight', 'conv_block4.conv2.k_weight',
                         'conv_block5.conv1.r_weight', 'conv_block5.conv1.i_weight', 'conv_block5.conv1.j_weight', 'conv_block5.conv1.k_weight',
                         'conv_block5.conv2.r_weight', 'conv_block5.conv2.i_weight', 'conv_block5.conv2.j_weight', 'conv_block5.conv2.k_weight',
                         'conv_block6.conv1.r_weight', 'conv_block6.conv1.i_weight', 'conv_block6.conv1.j_weight', 'conv_block6.conv1.k_weight',
                         'conv_block6.conv2.r_weight', 'conv_block6.conv2.i_weight', 'conv_block6.conv2.j_weight', 'conv_block6.conv2.k_weight']

        prev_conv_key_list = ['conv_block1.conv1.r_weight', 'conv_block1.conv1.i_weight', 'conv_block1.conv1.j_weight', 'conv_block1.conv1.k_weight',
                              'conv_block1.conv2.r_weight', 'conv_block1.conv2.i_weight', 'conv_block1.conv2.j_weight', 'conv_block1.conv2.k_weight',
                              'conv_block2.conv1.r_weight', 'conv_block2.conv1.i_weight', 'conv_block2.conv1.j_weight', 'conv_block2.conv1.k_weight',
                              'conv_block2.conv2.r_weight', 'conv_block2.conv2.i_weight', 'conv_block2.conv2.j_weight', 'conv_block2.conv2.k_weight',
                              'conv_block3.conv1.r_weight', 'conv_block3.conv1.i_weight', 'conv_block3.conv1.j_weight', 'conv_block3.conv1.k_weight',
                              'conv_block3.conv2.r_weight', 'conv_block3.conv2.i_weight', 'conv_block3.conv2.j_weight', 'conv_block3.conv2.k_weight',
                              'conv_block4.conv1.r_weight', 'conv_block4.conv1.i_weight', 'conv_block4.conv1.j_weight', 'conv_block4.conv1.k_weight',
                              'conv_block4.conv2.r_weight', 'conv_block4.conv2.i_weight', 'conv_block4.conv2.j_weight', 'conv_block4.conv2.k_weight',
                              'conv_block5.conv1.r_weight', 'conv_block5.conv1.i_weight', 'conv_block5.conv1.j_weight', 'conv_block5.conv1.k_weight',
                              'conv_block5.conv2.r_weight', 'conv_block5.conv2.i_weight', 'conv_block5.conv2.j_weight', 'conv_block5.conv2.k_weight',
                              'conv_block6.conv1.r_weight', 'conv_block6.conv1.i_weight', 'conv_block6.conv1.j_weight', 'conv_block6.conv1.k_weight',
                              'conv_block6.conv2.r_weight', 'conv_block6.conv2.i_weight', 'conv_block6.conv2.j_weight', 'conv_block6.conv2.k_weight']


        Z = OrderedDict()
        j = 0
        i = 0
        bn_i =0
        for key in conv_key_list:
            W_2D = weights[key].numpy()
            # print("i, key, W_2d,conv_index,weights_pruned {}".format(i), key, W_2D.shape,conv_index[key].shape ,weights_pruned[key].shape)
            if i <= 3:
                weights_pruned[key] = torch.tensor(W_2D[conv_index[key],:,:,:])
            else:
                weights_pruned[key] = torch.tensor(W_2D[conv_index[key],:,:,:][:,conv_index[prev_conv_key_list[i-4]],:,:])

            if i%4 == 0:
                # print(key,bn_key_list[j])
                # print(conv_key_list[i],conv_key_list[i+1],conv_key_list[i+2],conv_key_list[i+3]) 

                # print(weights[bn_key_list[j]].shape)
                # print(conv_index[key])
                # print(weights[bn_key_list[j]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+1]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+2]][conv_index[key]].shape)
                # print(weights[bn_key_list[j+3]][conv_index[key]].shape)
                weights_pruned[bn_key_list[j]] = torch.cat((weights[bn_key_list[j]][conv_index[conv_key_list[i] ]], weights[bn_key_list[j]][conv_index[conv_key_list[i+1]]],
                                                            weights[bn_key_list[j]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j]][conv_index[conv_key_list[i+3]]]),dim=0) #weights[bn_key_list[j]][conv_index[key]]

                weights_pruned[bn_key_list[j+1]] =torch.cat((weights[bn_key_list[j+1]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+1]][conv_index[conv_key_list[i+1]]],
                                                             weights[bn_key_list[j+1]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+1]][conv_index[conv_key_list[i+3]]]), dim =0)

                weights_pruned[bn_key_list[j+2]] =torch.cat((weights[bn_key_list[j+2]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+2]][conv_index[conv_key_list[i+1]]],
                                                             weights[bn_key_list[j+2]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+2]][conv_index[conv_key_list[i+3]]]), dim=0)

                weights_pruned[bn_key_list[j+3]] = torch.cat((weights[bn_key_list[j+3]][conv_index[conv_key_list[i]]],weights[bn_key_list[j+3]][conv_index[conv_key_list[i+1]]],
                                                              weights[bn_key_list[j+3]][conv_index[conv_key_list[i+2]]],weights[bn_key_list[j+3]][conv_index[conv_key_list[i+3]]]), dim=0)

                j = j + 4



            i = i + 1
            # print(key)
            # weights[key] = torch.tensor(W_2D[0:50,:,:,:])
            filename = path + key + '.npy'
            # print("i, key, conv_index,weights_pruned {}".format(i), key, W_2D.shape,conv_index[key].shape ,weights_pruned[key].shape)
            # print(filename)
            # np.save(filename,sorted_index)
            # print(len(sorted_index))


        weights_pruned['fc1.r_weight'] = weights['fc1.r_weight'][conv_index['conv_block6.conv2.r_weight'],:]
        weights_pruned['fc1.i_weight'] = weights['fc1.i_weight'][conv_index['conv_block6.conv2.i_weight'],:]
        weights_pruned['fc1.j_weight'] = weights['fc1.j_weight'][conv_index['conv_block6.conv2.j_weight'],:]
        weights_pruned['fc1.k_weight'] = weights['fc1.k_weight'][conv_index['conv_block6.conv2.k_weight'],:]

        self.load_state_dict(weights_pruned)


    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)


        x_first = torchaudio.functional.compute_deltas(x)
        x_second = torchaudio.functional.compute_deltas(x_first)
        x_third = torchaudio.functional.compute_deltas(x_second)
        #quternionic converter
        x_quaternion = torch.cat([x,x_first, x_second, x_third], dim=1) # (batch_size, 4, time_steps, freq_bins)
        #print("spectrogram shape", x_quaternion.shape)
        x = self.logmel_extractor(x_quaternion)
        # x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        # clipwise_output = torch.log_softmax(self.fc_audioset(x))        
        # clipwise_output = nn.functional.softmax(self.fc_audioset(x))        

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict    
