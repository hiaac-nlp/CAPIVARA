# <img src="assets/capivara.png" style="width:50px; margin-right:-5px"> CAPIVARA: Cost-Efficient Approach for Improving Multilingual CLIP Performance on Low-Resource Languages

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


|Model| Flickr30k
## Reproducibility

### Installation Requirements
Run the following command to install required packages.

```bash
pip install -r requirements.txt
```

### Code organization

### Data preprocessing

### Train

### Inference
In order to make easier to replicate our experiments, we share the scripts we used for inference.

#### Zero-shot cross-modal retrieval


#### Zero-shot image classification


## Citation
```bibtex
@inproceedings{capivara,
  title={CAPIVARA: Cost-Efficient Approach for Improving Multilingual CLIP Performance on Low-Resource Languages},
  ...
}
`