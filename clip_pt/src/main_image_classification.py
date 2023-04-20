import argparse
import logging
import os

import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, CLIPFeatureExtractor

from models.clip_pt_br_wrapper_image_classification import CLIPPTBRWrapperImageClassification
from utils.dataset.load_datasets import load_datasets

from utils.carbon_tracker import carbon_tracker_init,carbon_tracker_end
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level='ERROR')
load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        default=os.path.join("../", "experiment_setup", "warmup_finetuning.yaml"),
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument("-g", "--gpu", required=True, type=int)

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    vision_processor = CLIPFeatureExtractor.from_pretrained(config.model.image_encoder,
                                                            cache_dir='/hahomes/gabriel.santos/')
    text_tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder,
                                                   do_lower_case=False,
                                                   cache_dir='/hahomes/gabriel.santos/')

    datasets = load_datasets(config=config,
                             vision_processor=vision_processor,
                             text_tokenizer=text_tokenizer)

    train_dataloader = datasets["train_dataloader"]
    val_dataloader = [datasets["val_dataloader"], datasets["img_classification"]]
    train_size = datasets["train_size"]

    tracker_code_carbon = carbon_tracker_init(tracking_mode=config.carbon["process"], gpu_ids=[args.gpu])

    clip_pt = CLIPPTBRWrapperImageClassification(config, train_size,
                                                 val_labels=datasets["img_classif_labels"],
                                                 carbon_tracker=tracker_code_carbon)

    logger = WandbLogger(project="CLIP-PT", name=config.title)
    config["model_checkpoint"].pop("dirpath")

    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[
            ModelCheckpoint(**config["model_checkpoint"]),
            LearningRateMonitor("step")
        ],
        devices=[args.gpu],
        default_root_dir=os.path.join("../checkpoints/clip_pt", config["title"])
    )
    trainer.fit(clip_pt, train_dataloader, val_dataloader)

    carbon_tracker_end(tracker_code_carbon)

if __name__ == "__main__":
    main()

