import torch
import torch.nn as nn

from models.open_CLIP import OpenCLIP


class MultiHeadSimilarity(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSimilarity, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key):
        batch_size = query.size(0)

        # Linear transformations
        query = self.query_linear(query)  # Batch x Dim
        key = self.key_linear(key)  # Batch x Dim

        # Reshape query, key, and value
        query = query.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)  # Heads x Batch x HDim
        key = key.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1) # Heads x Batch x HDim

        # Scaled Dot-Product
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).to(query.device)) # Heads x Batch x Batch
        return scores.mean(dim=0)  # Batch x Batch


class MultiHeadCLIP(OpenCLIP):
    def __init__(self):
        super().__init__()
        self.similarity_block = MultiHeadSimilarity(d_model=512, num_heads=8)

        for param in self.model.visual.parameters():
            param.requires_grad = False

        for param in self.model.text.parameters():
            param.requires_grad = False

    def compute_logits(
            self,
            image_features,
            text_features,
            **kwargs
    ):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits_per_text = self.similarity_block(query=text_features, key=image_features)
        logits_per_image = logits_per_text.t()

        return logits_per_image, logits_per_text
