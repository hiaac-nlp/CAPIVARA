import argparse
import logging
import os

import pytorch_lightning as pl
import wandb
from codecarbon import EmissionsTracker
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, CLIPTokenizerFast

from models.teacher_student_clip_pt_br_wrapper import TeacherStudentCLIPPTBRWrapper
from utils.dataset.load_datasets import load_datasets_teacher_student

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

    teacher_tokenizer = CLIPTokenizerFast.from_pretrained(config.model.teacher,
                                                          cache_dir='/hahomes/gabriel.santos/')
    student_tokenizer = AutoTokenizer.from_pretrained(config.model.student,
                                                      do_lower_case=False,
                                                      cache_dir='/hahomes/gabriel.santos/')

    datasets = load_datasets_teacher_student(config=config,
                                             teacher_tokenizer=teacher_tokenizer,
                                             student_tokenizer=student_tokenizer)

    train_dataloader = datasets["train_dataloader"]
    val_dataloader = datasets["val_dataloader"]
    train_size = datasets["train_size"]

    # tracker_code_carbon = EmissionsTracker(log_level='error',
    #                                        tracking_mode=config.carbon["process"],
    #                                        gpu_ids=[args.gpu])
    # tracker_code_carbon.start()

    clip_pt = TeacherStudentCLIPPTBRWrapper(config, train_size,
                                            carbon_tracker=None)

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
        default_root_dir=os.path.join("../checkpoints/clip_pt_teacher_student", config["title"])
    )
    trainer.fit(clip_pt, train_dataloader, val_dataloader)

    # our_emission = tracker_code_carbon.flush()
    # our_energy = tracker_code_carbon._total_energy.__float__()
    # tracker_code_carbon.stop()

    # wandb.log({"carbon/Final Emission (CodeCarbon)": our_emission})
    # wandb.log({"carbon/Final Emission": our_energy * config.carbon["brazil_carbon_intensity"]})
    # wandb.log({"carbon/Final Energy": our_energy})


if __name__ == "__main__":
    main()
