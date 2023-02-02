import argparse
import logging
import os

import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from transformers import CLIPProcessor, AutoTokenizer

from utils.dataset.load_datasets import load_datasets
from models.clip_pt_br_wrapper import CLIPPTBRWrapper

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
    parser.add_argument("-g", "--gpu", required=True, type=str,)

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                                     cache_dir='/hahomes/gabriel.santos/')
    text_tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased',
                                                   do_lower_case=False,
                                                   cache_dir='/hahomes/gabriel.santos/')

    train_dataloader, val_dataloader = load_datasets(config=config,
                                                     vision_processor=vision_processor,
                                                     text_tokenizer=text_tokenizer)

    clip_pt = CLIPPTBRWrapper(config.model)
    neptune_logger = NeptuneLogger(
        project=os.environ.get("NEPTUNE_PROJECT"),
        api_token=os.environ.get("NEPTUNE_API_TOKEN"),
    )

    trainer = pl.Trainer(
        **config["trainer"],
        logger=neptune_logger,
        callbacks=[
            ModelCheckpoint(**config["model_checkpoint"]),
            EarlyStopping(**config["early_stopping"]),
            LearningRateMonitor("step")
        ],
        devices=args.gpu
    )
    trainer.fit(clip_pt, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()

