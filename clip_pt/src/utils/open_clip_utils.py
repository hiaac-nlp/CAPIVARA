def tokenize(example, text_tokenizer, vision_processor, lang="pt"):
    if lang.lower() == "pt":
        caption = example[1]["captions-pt"][0]
    else:
        caption = example[1]["captions-en"][0]

    text_input = text_tokenizer(caption)
    image_input = vision_processor(example[0])

    return image_input, text_input, example


def format_batch(batch):
    image_input = batch[0]
    text_input = batch[1].reshape((-1, 77))
    return image_input, text_input, batch[2]


def compute_similarity(model, batch, device, return_diag=True):
    image_input, text_input, examples = batch
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    batch = image_input, text_input

    img_features, txt_features = model(batch)

    norm_img_features = img_features / img_features.norm(dim=1, keepdim=True)
    norm_txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)

    sim = norm_txt_features @ norm_img_features.T

    if return_diag:
        return sim.diag()  # similarity between corresponding texts and images
    else:
        return sim