import os
import argparse

import logging
logging.basicConfig(level='ERROR')

import pandas as pd
import neptune.new as neptune
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from models.clip_pt_br_wrapper import CLIPPTBRWrapper
from utils.utils import prepare_pracegover, prepare_image_text_dataloader

from dotenv import load_dotenv
load_dotenv()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        default=os.path.join("../", "experiment_setup", "baseline.yaml"),
        type=str,
        help="YAML file with configurations"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    df_train, df_val = prepare_pracegover(
        train_metadata_path="/datasets/pracegover/pracegover_173k/pracegover_dataset.json",
        val_metadata_path="/datasets/pracegover/pracegover_173k/pracegover_captions_val2014.json"
    )

    train_loader, val_loader = prepare_image_text_dataloader(
        df_train,
        df_val,
        config
    )

    clip_pt = CLIPPTBRWrapper(config.model)
    neptune_logger = NeptuneLogger(
        project=os.environ.get("NEPTUNE_PROJECT"),
        api_token=os.environ.get("NEPTUNE_API_TOKEN"),
    )

    trainer = pl.Trainer(
        # devices=-1 ,strategy="ddp_sharded", # strategy="ddp_sharded"
        logger=neptune_logger,
        **config["trainer"],
        callbacks=[
            ModelCheckpoint(
                **config["model_checkpoint"]
            ),
            EarlyStopping(
                **config["early_stopping"]
            ),
            LearningRateMonitor("step")
        ]
    )
    trainer.fit(clip_pt, train_loader, val_loader)

if __name__ == "__main__":
    main()