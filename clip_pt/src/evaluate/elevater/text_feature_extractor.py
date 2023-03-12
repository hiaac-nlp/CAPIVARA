import tqdm
import torch

from elevater.resources.translated_prompts import class_map, template_map


def extract_text_features(dataset_name: str, model, text_tokenizer, device):
    model.to("cpu")
    class_names = class_map.get(dataset_name)

    templates = template_map.get(dataset_name, ["a foto de um {}"])

    zeroshot_weights = []

    for classname in tqdm.tqdm(class_names):
        # if type(classname) == list: classname = classname[0]
        texts = [template.format(classname) for template in templates]

        tokenized_text = text_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=95
        )

        class_embeddings = model.encode_text(tokenized_text)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.extend(class_embedding.unsqueeze(0))

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)

    return zeroshot_weights