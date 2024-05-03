import numpy
import os
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import vector_quantize_pytorch
from wavlm.WavLM import WavLM, WavLMConfig

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the pre-trained checkpoints
checkpoint = torch.load('./WavLM-Base+.pt')
cfg = WavLMConfig(checkpoint['cfg'])
wavlm_model = WavLM(cfg).to(device)
wavlm_model.load_state_dict(checkpoint['model'])
wavlm_model.eval()


def load_data():
    items = []
    count = 0
    for f in os.listdir("./speech_samples"):
        data, samplerate = sf.read(os.path.join("./speech_samples/", f))
        data = numpy.reshape(data, [1, len(data)])
        wav_input = torch.from_numpy(data).to(device)
        wav_input = wav_input.float()
        with torch.no_grad():
            rep = wavlm_model.extract_features(wav_input)[0]
        #            rep_shape = rep.shape
        #            rep = numpy.reshape(rep, (rep_shape[1], rep_shape[2]))
        items.append((data, samplerate, rep))
        count += 1
        if count > 10:
            break

    max_size = 0
    min_size = 1000000
    for item in items:
        cur_size = item[2].shape[1]
        max_size = max(max_size, cur_size)
        min_size = min(min_size, cur_size)

    items2 = []
    count = 0
    for data, framerate, features in items:
        features2 = F.pad(features, (0, 0, max_size - features.shape[1], 0), "constant", 0)
        items2.append((features2, torch.tensor([count])))
        count += 1
    return items2


class LSTMPhonemeClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, codebook_size=512):
        super(LSTMPhonemeClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.num_classes = num_classes

        self.vq = vector_quantize_pytorch.VectorQuantize(
            dim=self.input_dim,
            codebook_size=self.codebook_size,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.,  # the weight on the commitment loss
            ema_update=False,
            straight_through=True,
            reinmax=True,
            learnable_codebook=True
        )

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.codebook_size, self.hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2phoneme = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x):
        x_shape = x.shape
        assert x_shape[2] == self.input_dim
        assert x_shape[0] > 0
        hidden = (torch.randn(1, self.hidden_dim), torch.randn(1, self.hidden_dim))
        _, indices, _ = self.vq(x)
        indices = numpy.reshape(indices, (indices.shape[1]))
        onehot = F.one_hot(indices, self.codebook_size)
        onehot = onehot.float()
        out, hidden = self.lstm(onehot, hidden)
        phoneme_space = self.hidden2phoneme(hidden[0])
        phoneme_scores = F.log_softmax(phoneme_space, dim=1)
        return phoneme_scores


def train_lstm(model: LSTMPhonemeClassifier, data, epochs):
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        for datum, target_phoneme in data:
            model.zero_grad()

            phoneme_scores = model(datum)
            loss = loss_function(phoneme_scores, target_phoneme)
            loss.backward()
            optimizer.step()


training_data = load_data()
pc = LSTMPhonemeClassifier(768, 256, 1584)

with torch.no_grad():
    tag_scores = pc(training_data[0][0])
    print(max(tag_scores[0]))

train_lstm(pc, training_data, 2)

with torch.no_grad():
    tag_scores = pc(training_data[0][0])
    print(max(tag_scores[0]))