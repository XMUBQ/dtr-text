import openai
import torch.nn as nn
keys='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'
openai.api_key = keys# os.getenv(keys)
prompts="Decide whether a paragraph's quality is good or bad.\n\nParagraph: [text]\nQuality:"

def flatten(l):
    return [item for sublist in l for item in sublist]

def batch(iterable, n=16):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def get_quality(sents):
    y=[]
    for sent in sents:
        response = openai.Completion.create(
        engine="text-curie-001",
        prompt=prompts.replace('[text]',sent),
        temperature=0.7,
        max_tokens=16,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        y.append(response['choices'][0].text.strip())
    return y

def quality2num(label):
    num_label=-1
    if 'good' in label.lower() or 'decent' in label.lower():
        num_label=1
    elif 'bad' in label.lower() or 'poor' in label.lower():
        num_label=0
    else:
        print(label)
    return num_label