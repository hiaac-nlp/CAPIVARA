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
from models.self_distill_clip_wrapper import SelfDistillCLIPWrapper, \
    TeacherStudentSelfDistillCLIPWrapper
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
        default=os.path.join("../", "experiment_setup", "warmup_finetuning.yaml"),
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument("-g", "--gpu", required=True, type=int)

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    if config.get("model", None) is None:
        # model doesn't have adapters
        model = OpenCLIP()
    else:
        # model has adapters
        model = OpenCLIPAdapter(adapter=config.model.adapter, devices=args.gpu)

    vision_processor = model.image_preprocessor
    text_tokenizer = model.text_tokenizer

    datasets = load_datasets(config=config,
                             vision_processor=vision_processor,
                             text_tokenizer=text_tokenizer)

    train_dataloader = datasets["train_dataloader"]
    val_dataloader = [datasets["val_dataloader"], datasets["img_classification"]]
    train_size = datasets["train_size"]

    tracker_code_carbon = carbon_tracker_init(tracking_mode=config.carbon["process"],
                                              gpu_ids=[args.gpu],
                                              carbon_checker=config.carbon["carbon_checker"])

    if config.get("self_distill", False) == "teacher":
        clip_pt = TeacherStudentSelfDistillCLIPWrapper(config, train_size,
                                                       val_labels=datasets["img_classif_labels"],
                                                       model=model,
                                                       carbon_tracker=tracker_code_carbon)
    elif config.get("self_distill", False):
        clip_pt = SelfDistillCLIPWrapper(config, train_size,
                                         val_labels=datasets["img_classif_labels"],
                                         model=model,
                                         carbon_tracker=tracker_code_carbon)
    else:
        clip_pt = OpenCLIPWrapper(config, train_size,
                                  val_labels=datasets["img_classif_labels"],
                                  model=model,
                                  carbon_tracker=tracker_code_carbon)

    wandb.init(project="CLIP-PT", name=config.title)
    logger = WandbLogger(project="CLIP-PT", name=config.title)
    config["model_checkpoint"].pop("dirpath")

    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[
            ModelCheckpoint(**config["model_checkpoint"]),
            LearningRateMonitor("step"),
            AdaptersActivation(config.model.number_layers,config.model.progressive_adapter),
        ],
        devices=[args.gpu],
        default_root_dir=os.path.join("../checkpoints/open_clip_pt", config["title"])
    )
    trainer.fit(clip_pt, train_dataloader, val_dataloader)

    if config.get("model", None) is not None:
        # model has adapters
        print('saving the adapters')
        clip_pt.model.model.text.save_all_adapters(f"CLIP-PT/adapter_checkpoints/{wandb.run.id}")

    carbon_tracker_end(tracker_code_carbon, config.carbon["brazil_carbon_intensity"])


if __name__ == "__main__":
    main()
