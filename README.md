# <img src="assets/capivara.png" style="width:50px; margin-right:-5px"> CAPIVARA: Cost-Efficient Approach for Improving Multilingual CLIP Performance on Low-Resource Languages

[![Arxiv](http://img.shields.io/badge/Arxiv-2023-B31B1B.svg)](https://arxiv.org/abs/2310.13683)

In this project, we propose <img src="assets/capivara.png" style="width:20px"> CAPIVARA, a cost-efficient framework
designed to enhance the performance of multilingual CLIP models in low-resource languages. Our framework are built upon
pre-trained [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main#openclip), and it implements the
conventional fine-tuning and also an optimized fine-tuning (CAPIVARA + Opt.) that uses LoRA and gradient checkpointing in order to reduce 
the computation cost.

<img src="assets/capivara.png" style="width:20px"> CAPIVARA holds the state of the art in many zero-shot tasks involving 
images and Portuguese texts. Also, our method has the potential of significantly improve the model performance in other 
low-resource languages using a single RTX Quadro 8000 GPU for just 2 hours.

## Pipeline
<img src="assets/pipeline.png" >

In our pipeline, we employed the following models:

+ **Translator**: Google Translate
+ **Image captioning**: [BLIP2](https://huggingface.co/Salesforce/blip2-opt-2.7b-coco)

## Results

| <img src="assets/low-resource-lang.png" >                                                                                                                                                                                                                                                                                                                                                                       |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Performance improvement with CAPIVARA + Opt. in Low-Resource Languages: Xhosa, Hindi, and Portuguese. The percentage point increase over the baseline ([OpenCLIP ViT-B/32 XLM-Roberta Base](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k)) in terms of mean recall for text-to-image (txt2img) and image-to-text (img2txt) retrieval is highlighted above the respective bars. |


#### Zero-shot image classification

|                         | **Caltech-101** | **CIFAR-10** | **CIFAR-100** | **Country-211** | **DTD**      | **EuroSAT**  | **FER-2013** | **FGVC-Aircraft** | **Food-101** | **GTSRB**    | **Hateful-Memes** | **KITTI-Distance** | **MNIST**    | **Oxford Flowers-102** | **Oxford-IIIT Pets** | **PatchCamelyon** | **Rendered-SST2** | **RESISC-45** | **Stanford-Cars** | **PASCAL VOC-2007** | **Average**  | **ImageNet-1k** |
|:-----------------------:|:---------------:|:------------:|:-------------:|:---------------:|:------------:|:------------:|:------------:|:-----------------:|:------------:|:------------:|:-----------------:|:------------------:|:------------:|:----------------------:|:--------------------:|:-----------------:|:-----------------:|:-------------:|:-----------------:|:-------------------:|:------------:|:---------------:|
| **OpenCLIP (Baseline)** | 84.53 ± 0.00    | 93.99 ± 0.00 | 68.44 ± 0.00  | 17.82 ± 0.00    | 41.17 ± 0.00 | 47.16 ± 0.00 | 48.65 ± 0.00 | 26.30 ± 0.00      | 65.06 ± 0.00 | 43.27 ± 0.00 | 56.50 ± 0.00      | 28.41 ± 0.00       | 54.99 ± 0.00 | 50.88 ± 0.00           | 81.56 ± 0.00         | 50.96 ± 0.00      | 54.20 ± 0.00      | 58.51 ± 0.00  | 84.93 ± 0.00      | 82.09 ± 0.00        | 56.97 ± 0.00 | 45.84 ± 0.00    |
|      **CAPIVARA**       | 82.97 ± 0.03    | 93.85 ± 0.00 | 69.37 ± 0.01  | 17.61 ± 0.00    | 42.34 ± 0.04 | 47.77 ± 0.02 | 46.68 ± 0.05 | 25.49 ± 0.01      | 64.58 ± 0.01 | 46.34 ± 0.01 | 56.17 ± 0.00      | 33.94 ± 0.13       | 60.14 ± 0.04 | 49.93 ± 0.02           | 79.37 ± 0.00         | 51.71 ± 0.01      | 54.82 ± 0.03      | 59.71 ± 0.01  | 85.10 ± 0.02      | 82.29 ± 0.00        | **57.51 ± 0.02** | **46.06 ± 0.01**    |
|   **CAPIVARA + Opt.**   | 83.68 ± 0.02    | 93.93 ± 0.03 | 68.87 ± 0.01  | 17.32 ± 0.02    | 41.79 ± 0.07 | 48.85 ± 0.12 | 46.85 ± 0.13 | 25.54 ± 0.09      | 64.46 ± 0.00 | 44.66 ± 0.06 | 56.81 ± 0.03      | 28.27 ± 0.11       | 55.00 ± 0.10 | 51.99 ± 0.12           | 80.90 ± 0.09         | 52.39 ± 0.07      | 52.94 ± 0.04      | 56.93 ± 0.01  | 84.90 ± 0.06      | 81.99 ± 0.02        | 56.90 ± 0.06 | 45.65 ± 0.02    |


## Reproducibility

### Installation
Run the following command to install required packages.

```bash
pip install -r requirements.txt
```

### Code organization

```
├─ README.md
├─ assets
│  ├─ capivara.png
│  ├─ low-resource-lang.png
│  └─ pipeline.png
├─ clip_pt
│  ├─ experiment_setup					<--- training setup files in format .yaml			
│  │  └─ capivara.yaml
│  ├─ requirements.txt
│  └─ src
│     ├─ evaluate
│     │  ├─ utils
│     │  │  ├─ metric.py				<--- metrics used in ELEVATOR
│     │  │  ├─ voc2007.py				<--- metric used in PASCAL VOC-2007
│     │  │  └─ resources				<--- setup files used for inference
│     │  ├─ zero_shot_elevater.py			<--- script used for zero-shot image classification in ELEVATOR
│     │  ├─ zero_shot_image_classification.py		<--- script used for zero-shot image classification in ImageNet, ObjectNet, and GroceryStore
│     │  ├─ zero_shot_imagenet_babel.py			<--- script used for zero-shot image classification in ImageNet Babel
│     │  └─ zero_shot_retrieval.py			<--- script used for zero-shot cross modal retrieval
│     ├─ generating
│     │  ├─ blip2.py					<--- script used for generating synthetic captions
│     │  └─ generated_caps_sim_score.py			<--- script used for compute similarity score between captions and images
│     ├─ main_open_clip.py				<--- main script used for training
│     ├─ models
│     │  ├─ open_CLIP.py				<--- base CLIP class
│     │  ├─ open_CLIP_adapter.py			<--- CLIP + LoRA class
│     │  └─ open_clip_wrapper.py			<--- Wrapper that implements the training methods using pytorch-lightning
│     ├─ recipes					<--- auxiliar executable files
│     └─ utils
│        ├─ carbon_tracker.py				<--- methods used to estimate the carbon footprint
│        ├─ loss.py					<--- loss function
│        ├─ open_clip_utils.py				<--- implements auxiliar methods
│        ├─ scheduler.py				<--- implements learning rate schedulers
│        └─ dataset
│           ├─ evaluation_dataset.py			<--- base evaluation class
│           ├─ grocery_store_dataset.py			<--- implements grocery store evaluation class 
│           ├─ imagenet_dataset.py			<--- implements ImageNet evaluation class 
│           ├─ object_net.py				<--- implements ObjectNet evaluation class 
│           └─ load_datasets_open_clip.py		<--- methods to load train/val datasets

└─ preprocessing					<--- auxiliar dataset preprocessing methods
```


### Data preprocessing
#### Dataset translation

Since the texts used are translated from English into the target languages, if it is necessary to introduce new data in addition to the data provided by us, a new translation is required. We used Google Translate to do this. First, we extracted all the captions for each of the sets used. Then we translated the captions using the translator. Finally, we added all the translated captions to their original bases with the tag of the language used. All sets are kept in the original format of the bases to make it easier for users who already use them.

#### Generating synthetic captions
To generate the synthetic captions, you can run the following command:

```bash
python3 generating/blip2.py --dataset-path "your_webdataset/{00000..00999}.tar" --gpu 0 --batch 100
```

It uses BLIP2 to generate captions starting with the prefixes:
```
prefixes = ["the foreground features", "a photo of", "a picture of",
            "this is a scene depicting", "an image of", "portrait of a",
            "this image captures a moment of", "a painting of", "an art of",
            "the picture shows"]
```

Then, a new dataset is saved whose name is ```"dataset_name_{postfix-path}"```, where ```--postfix-path``` 
is an optional argument.

### Train

CAPIVARA is built using [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/). The file [example.yaml](https://github.com/hiaac-nlp/CAPIVARA/blob/main/clip_pt/experiment_setup/example.yaml) lists all the parameters that can be used by CAPIVARA.

For simple and straightforward training of the model, the following command can be used:
```bash
python3 main_open_clip.py --config_path=path/to/config_file
```
To use the adapter training settings, you must also pass on the directory of the checkpoint used:
```bash
python3 main_open_clip.py \
		--config_path=path/to/config_file \
		--checkpoint-dir=path/to/checkpoint 
```
Other settings (all present in the file [example.yaml](https://github.com/hiaac-nlp/CAPIVARA/blob/main/clip_pt/experiment_setup/example.yaml) are available to configure the training and import according to your needs.

### Inference
In order to make easier to replicate our experiments, we share the scripts we used for inference.

#### Zero-shot Text-to-Image Retrieval

The following method can be used to retrieve images:

```python
def text_to_image_retrieval(text_required, model, image_features, text_features, all_images, all_texts):
    all_texts = sum(all_texts, [])
    caption = []
    for text in text_required:
        if type(text) != int:
            caption.append(text)
            text_features = text_tokenizer(text)
            text_features = model.encode_text(text_features.to(device))
            text_features = text_features
        else:
            caption.append([text])
        similarities = []
        for i in tqdm.tqdm(range(len(image_features)), desc="t2i retrieval"):
            if type(text) == int:
                scores = text_features[text] @ image_features[i].t()  # shape: [batch_size, batch_size]
            else:
                scores = text_features @ image_features[i].t()  # shape: [batch_size, batch_size]
            item = {
                'score': scores.cpu(),
                'id': i,
                'image': all_images[i].cpu()
            }
            similarities.append(item)
        similarities_df = pd.DataFrame(similarities)
        sorted_similarities_df = similarities_df.sort_values(by='score', ascending=False)
    return sorted_similarities_df, caption
```

In this way, a list containing the similarity scores between the input text and the set of images is returned, as well as their ids and images.

#### Zero-shot Image-to-Text Retrieval
As a complement, the method below retrieves text from a target image.

```python
def image_to_text_retrieval(image_required, image_features, text_features, all_images, all_texts):
    all_texts = sum(all_texts, [])
    images_selected = []
    for image in image_required:
        images_selected.append(all_images[image])
        similarities = []
        for i in tqdm.tqdm(range(len(text_features)), desc="i2t retrieval"):
            scores = text_features[i] @ image_features[image].t()  # shape: [batch_size, batch_size]
            item = {
                'score': scores.cpu(),
                'id': i,
                'text': all_texts[i]
            }
            similarities.append(item)
        similarities_df = pd.DataFrame(similarities)
        sorted_similarities_df = similarities_df.sort_values(by='score', ascending=False)
    return sorted_similarities_df, images_selected
```

This method returns a list containing the similarity scores between the input image and the set of texts, as well as their ids and images.
The use of these methods and other auxiliary methods can also be seen in the [retrieval example notebook](https://github.com/hiaac-nlp/CAPIVARA/blob/main/clip_pt/src/evaluate/capivara_retrieval.ipynb), where it is possible to iteratively retrieve images and texts.

#### Retrieval Evaluation

To carry out the evaluation of image and text retrieval automatically, generating the metrics used in the article, the python script [zero_shot_retrieval.py](https://github.com/hiaac-nlp/CAPIVARA/blob/main/clip_pt/src/evaluate/zero_shot_retrieval.py) can be used.
The following parameters can be used:

```bash
--model-path # directs to the path of the model checkpoint
--dataset-path # path to validation/test dataset
--translation # select which translation framework will be used "english", "marian", "google" (default)
--language # language used for captions: "en" (default), "xh", "hi"
--batch # batch size
--open_clip # indicates whether model is fine-tuned (True) or is the original OpenCLIP (False)
--gpu # select GPU
--adapter # load the adapter weights
```

#### Zero-shot image classification
To use the model as a classifier, the following code can be used:

```python
img_features, txt_features = model.model(batch)
logits, _ = model.model.compute_logits(
                img_features,
                txt_features,
                fixed_logit=False
            )  # shape: [n_imgs, n_classes]
predictions = torch.argsort(logits, descending=True)
predicted_labels = predictions[:, :k]

# Check if the target label is in the top-k predictions for each sample
correct_predictions = (predicted_labels == targets.view(-1, 1)).any(dim=1)

```

The predictions return the correct predictions relating the classified image and text. We then check the first k correctly classified values.
An [classification example notebook](https://github.com/hiaac-nlp/CAPIVARA/blob/main/clip_pt/src/evaluate/capivara_classification.ipynb) for classifying images and text is also available.

## Acknowledgements

This project was supported by the Ministry of Science, Technology, and Innovation of Brazil, with
resources granted by the Federal Law 8.248 of October 23, 1991, under the PPI-Softex. The project
was coordinated by Softex and published as Intelligent agents for mobile platforms based on
Cognitive Architecture technology [01245.013778/2020-21]. D.A.B.M. is partially funded by FAPESP 2023/05939-5.
A.I.F., T.S., N.S. are partially funded by Centro de Excelência em Inteligência Artificial (CEIA),
da Universidade Federal de Goiás (UFG). E.L.C. is partially funded by CNPq 315468/2021-1.
H.P. is partially funded by CNPq 304836/2022-2. S.A. is partially funded by CNPq 315231/2020-3,
FAPESP 2013/08293-7, 2020/09838-0, Google Award for Inclusion Research 2022.

## Citation
```bibtex
@inproceedings{santos2023capivara,
  title={CAPIVARA: Cost-Efficient Approach for Improving Multilingual CLIP Performance on Low-Resource Languages},
  author={Santos, Gabriel O. dos and Moreira, Diego A. B. and Ferreira, Alef I. and Silva, Jhessica and Pereira, Luiz and Bueno, Pedro and Sousa, Thiago and Maia, Helena and da Silva, N{\'a}dia and Colombini, Esther and Pedrini, Helio and Avila, Sandra},
  booktitle = " Workshop on Multi-lingual Representation Learning (MRL), Conference on Empirical Methods in Natural Language Processing (EMNLP)",
  year = "2023"
}
```
