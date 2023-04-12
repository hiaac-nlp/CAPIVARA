import argparse
import logging
import os

import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, CLIPFeatureExtractor

from models.clip_pt_br_wrapper import CLIPPTBRWrapper
from utils.dataset.load_datasets import load_datasets

from codecarbon import EmissionsTracker
import wandb

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

    vision_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32",
                                                            cache_dir='/hahomes/gabriel.santos/')
    text_tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased',
                                                   do_lower_case=False,
                                                   cache_dir='/hahomes/gabriel.santos/')

    datasets = load_datasets(config=config,
                             vision_processor=vision_processor,
                             text_tokenizer=text_tokenizer)

    train_dataloader = datasets["train_dataloader"]
    val_dataloader = datasets["val_dataloader"]
    train_size = datasets["train_size"]
    val_size = datasets["val_size"]

    tracker_code_carbon = EmissionsTracker(log_level = 'error', tracking_mode=config.carbon["process"], gpu_ids=[args.gpu])
    tracker_code_carbon.start()

    clip_pt = CLIPPTBRWrapper(config, train_size, val_size, carbon_tracker=tracker_code_carbon)
    logger = WandbLogger(project="CLIP-PT",
                         name=config.title)

    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[
            ModelCheckpoint(**config["model_checkpoint"]),
            LearningRateMonitor("step")
        ],
        devices=[args.gpu]
    )
    trainer.fit(clip_pt, train_dataloader, val_dataloader)

    our_emission = tracker_code_carbon.flush()
    our_energy = tracker_code_carbon._total_energy.__float__()
    tracker_code_carbon.stop()

    wandb.log({"carbon/Final Emission (CodeCarbon)": our_emission})
    wandb.log({"carbon/Final Emission": our_energy * config.carbon["brazil_carbon_intensity"]})
    wandb.log({"carbon/Final Energy": our_energy})


if __name__ == "__main__":
    main()

