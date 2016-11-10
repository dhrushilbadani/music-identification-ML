# Notes

Here are some papers worth referring to:

  - [Unsupervised Learning of Video Representations using LSTMs:](https://arxiv.org/pdf/1502.04681v3.pdf) 
    - "We use multilayer Long Short Term Memory
(LSTM) networks to learn representations of
video sequences. Our model uses an encoder
LSTM to map an input sequence into a fixed
length representation. This representation is decoded
using single or multiple decoder LSTMs
to perform different tasks, such as reconstructing
the input sequence, or predicting the future
sequence."
    - This is almost like word2vec for videos. We can try something similar for our dataset, either the raw audio or a mid-level feature like mel-spectrograms, MFCC etc.
  - [Translating Videos to Natural Language
Using Deep Recurrent Neural Networks](https://arxiv.org/pdf/1412.4729.pdf)   
    - "[...] we propose
to translate videos directly to sentences using
a unified deep neural network with both convolutional
and recurrent structure." 
    - They apply a ConvNet on each frame of the video for feature extraction, and apply mean pooling on these processed frames. They treat the pooled features for these frames as a video sequence, and feed the sequence to an LSTM-RNN. 
  - [Denoising Convolutional Autoencoders for Noisy Speech Recognition](http://cs231n.stanford.edu/reports/Final_Report_mkayser_vzhong.pdf)
    -  This paper is interesting as it first passes the raw audio through a ConvNet-based denoising autoencoder and then subsequently extracts the MFCC features from the denoised audio and then does its audio recognition. We could do something similar for the live-song recognition task - use an RNN/CNN-based autoencoder and then calculate its hashprints.
- [Semi-supervised Sequence Learning](https://arxiv.org/pdf/1511.01432v1.pdf)


