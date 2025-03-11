from typing import List, Optional, Tuple

import os
import sys
import hydra
import lightning as L
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from newsreclib import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from newsreclib import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: Configuration composed by Hydra.

    Returns:
        Dictionary with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # force download of the dataset, if not already downloaded in order to instantiate model with pretrained entity embeddings
    log.info("Downloading and parsing dataset, if not cached.")
    datamodule.prepare_data()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, limit_predict_batches=2)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("embed"):
        assert 'embed_save_path' in cfg, "Need to specify 'embed_save_path' in config"
        for split in ("train", "valid", "test"):
            log.info(f"Generating embeddings on {split}")
            datamodule.predict_dataloader = getattr(datamodule, f"{split}_dataloader")
            predictions = trainer.predict(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
            unique_user_embeddings = set()
            unique_news_embeddings = set()
            for i, (_, user, news, mask) in enumerate(predictions):
                # user: (batch size, embed dim)
                # news: (batch size, max # of articles, embed dim)
                # mask: (batch size, max # of articles)
                for u in user:
                    embedding = tuple(u.detach().cpu().numpy())
                    unique_user_embeddings.add(embedding)
                for news_batch, mask_batch in zip(news, mask):
                    for article, m in zip(news_batch, mask_batch):
                        if m.item() != 0:
                            embedding = tuple(article.detach().cpu().numpy())
                            unique_news_embeddings.add(embedding)
            user_embeds = torch.tensor(list(unique_user_embeddings))
            news_embeds = torch.tensor(list(unique_news_embeddings))
            save_path = cfg["embed_save_path"].replace("{exp}", cfg.experiment)
            save_path = save_path.replace("{split}", split)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'user_embeds': user_embeds, 'news_embeds': news_embeds}, save_path)
            log.info(f"Saved to {save_path}")
        sys.exit(0)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    
    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
