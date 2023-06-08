import json
from collections import defaultdict

from tqdm import tqdm
file_path = '/hadatasets/pracegover/pracegover_400k.json'
with open(file_path) as file:
    data = json.load(file)

splits = ['train', 'validation', 'test']

for split in splits:
    print(split)
    freq = defaultdict(lambda: 0)
    for example in tqdm(data[split]):
        l = len(example['caption'].split())
        freq[l] += 1
    print(freq)
