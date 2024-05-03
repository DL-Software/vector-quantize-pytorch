import espeak

es = espeak.ESpeak(voice="en-gb")

consonants = ["p", "b", "t", "d", "tS", "dZ", "k"]
consonants += ["g", "f", "v", "T", "D", "s", "z", "S"]
consonants += ["Z", "h", "m", "n", "N","l","r","j","w"]

vowels = ["@", "3", "3:", "@L", "@2", "@5", "a", "aa"]
vowels += ["a#", "A:", "A@", "E", "e@", "I", "I2", "i", "i:", "i@"]
vowels += ["0", "V", "u:", "U", "U@", "O:", "O@", "o@"]
vowels += ["aI", "eI", "OI", "aU", "oU", "aI@", "aU@"]

samples = []

for c in consonants:
    for v in vowels:
        samples.append("[[" + c + v + "]]")
        samples.append("[[" + v + c + "]]")

outpath_root = "./speech_samples/"
for x in range(len(samples)):
    outpath = outpath_root + str(x) + ".wav"
    es.save(samples[x], outpath)
