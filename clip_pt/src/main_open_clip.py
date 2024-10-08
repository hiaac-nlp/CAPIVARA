import argparse
import logging
import os

import pytorch_lightning as pl
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from models.open_CLIP import OpenCLIP
from models.open_CLIP_adapter import OpenCLIPAdapter
from models.open_clip_wrapper import OpenCLIPWrapper

from utils.carbon_tracker import carbon_tracker_init, carbon_tracker_end
from utils.dataset.load_datasets_open_clip import load_datasets
from utils.callbacks import AdaptersActivation

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level='ERROR')
load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument("-g", "--gpu", required=True, type=int)
    parser.add_argument("-ck", "--checkpoint-dir", required=False, type=str, default="../checkpoints/open_clip_pt")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    if "tags" not in config or config.tags is None:
        raise Exception(f"You must add a list of tags in attribute ``tags`` in your experiment \
                        setup file {args.config_path}.\n \
                        E.g.\n\
                        tags:\n\
                            - <your tag>")

    if config.get("model", None) is None:
        # model doesn't have adapters
        model = OpenCLIP()
    else:
        # model has adapters
        model = OpenCLIPAdapter(
            adapter=config.model.adapter,
            devices=args.gpu,
            projection_layer=config.model.projection_layer,
            load_pretrained_weights=config.get("load_pretrained_weights", False),
            path_to_pretrained_weights=config.get("path_to_pretrained_weights", None)
        )

    vision_processor = model.image_preprocessor
    text_tokenizer = model.text_tokenizer

    datasets = load_datasets(
        config=config,
        vision_processor=vision_processor,
        text_tokenizer=text_tokenizer
    )

    train_dataloader = datasets["train_dataloader"]
    val_dataloader = [datasets["val_dataloader"], datasets["img_classification"]]

    tracker_code_carbon = carbon_tracker_init(
        tracking_mode=config.carbon["process"],
        gpu_ids=[args.gpu],
        carbon_checker=config.carbon["carbon_checker"]
    )

    clip_pt = OpenCLIPWrapper(
        config,
        val_labels=datasets["img_classif_labels"],
        model=model,
        carbon_tracker=tracker_code_carbon,
        load_pretrained_weights=config.get("load_pretrained_weights", False),
        path_to_pretrained_weights=config.get("path_to_pretrained_weights", None)
    )

    tags = ["open_clip"]
    tags += [dataset["name"] for dataset in config.datasets.train]  # add training datasets as tags
    tags += config.tags  # add tags defined for experiments
    wandb.init(project="CLIP-PT", name=config.title, tags=tags)
    logger = WandbLogger(project="CLIP-PT", name=config.title, tags=tags)
    config["model_checkpoint"].pop("dirpath")

    callbacks = [
        ModelCheckpoint(**config["model_checkpoint"]),
        LearningRateMonitor("step"),
    ]
    if config.get("model", None) is not None:
        callbacks.append(AdaptersActivation(config.model.number_layers, config.model.progressive_adapter))

    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=callbacks,
        devices=[args.gpu],
        default_root_dir=os.path.join(args.checkpoint_dir, config["title"])
    )
    trainer.fit(clip_pt, train_dataloader, val_dataloader)

    if config.get("model", None) is not None:
        # model has adapters
        print('saving the adapters')
        clip_pt.model.model.text.save_pretrained(f"{args.checkpoint_dir}/adapter_PEFT_checkpoints/{wandb.run.id}")

    carbon_tracker_end(tracker_code_carbon, config.carbon["brazil_carbon_intensity"],carbon_checker=config.carbon["carbon_checker"])


if __name__ == "__main__":
    main()
