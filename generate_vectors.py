import numpy
import soundfile as sf
import os
import torch
import torch.nn.functional as F

import lstm_phoneme_classifier
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
from wavlm.WavLM import WavLM, WavLMConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the pre-trained checkpoints
checkpoint = torch.load('./WavLM-Base+.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

items = []
count = 0
for f in os.listdir("./speech_samples"):
    data, samplerate = sf.read(os.path.join("./speech_samples/", f))
    data = numpy.reshape(data, [1, len(data)])
    wav_input = torch.from_numpy(data).to(device)
    wav_input = wav_input.float()
    with torch.no_grad():
        rep = model.extract_features(wav_input)[0]
#        rep_shape = rep.shape
#        rep = numpy.reshape(rep, (rep_shape[1], rep_shape[2]))
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
for data, framerate, features in items:
    features2 = F.pad(features, (0, 0, max_size - features.shape[1], 0), "constant", 0)
    items2.append((data, framerate, features2))

rnn = torch.nn.RNN(10, 200)

print(len(items))
print(items[3])
print(min(items[3][0][0]))
print(items[3][2].shape)
print(items2[3][2].shape)
print(items[4][2].shape)
print(items2[4][2].shape)

#vq = VectorQuantize(
#            dim = 768,
#            codebook_size = 512,     # codebook size
#            decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
#            commitment_weight = 1.   # the weight on the commitment loss
#        )

#test_item = items2[0][2]

#quantized, indices, commit_loss = vq(test_item)
#print(indices)
#print(commit_loss)
#for x in range(10):
#    print(x)
#    for item in items2:
#        data = item[2]
#        quantized, indices, commit_loss = vq(data)

#quantized, indices, commit_loss = vq(test_item)
#print(indices)
#print(commit_loss)


pc = lstm_phoneme_classifier.LSTMPhonemeClassifier(768, 256, 1584)
scores = pc(items[3][2])

print(scores[0][0:20])
print(min(scores[0]))
print(max(scores[0]))
