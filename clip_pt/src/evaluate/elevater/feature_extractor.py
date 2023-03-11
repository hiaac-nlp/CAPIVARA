import time
import logging

import tqdm
import torch
import numpy as np
from transformers import BatchFeature

def extract_feature(data_loader, model, device):
    model.to(device)
    start = time.time()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            x, y = batch[:2]
            all_labels.append(y.cpu().numpy())
            if (isinstance(x, dict) or isinstance(x, BatchFeature)) \
                    and "pixel_values" in x:
                x["pixel_values"] = x["pixel_values"].squeeze(1).to(device)
            # compute output
            if device == torch.device('cuda'):
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
            outputs = model.encode_visual(x["pixel_values"])
            all_features.append(outputs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)
    logging.info(f'=> Feature extraction duration time: {time.time() - start:.2f}s')
    return np.reshape(features, (features.shape[0], -1)), np.reshape(labels, (labels.shape[0], -1))