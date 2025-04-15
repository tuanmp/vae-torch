from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning import callbacks
import os
import models
import data
import click
import yaml
import torch
from functools import partial

cache_dir = os.environ.get("PSCRATCH", "/tmp")

@click.command()
@click.option("--config", default="config/vae.yaml", help="Path to the config file.")
@click.option("--fast_dev_run", default=False, help="Run a fast dev run.")
def train(config, fast_dev_run):
    
    # load config file
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    
    # load model
    model = getattr(models, config['model']['model_class'])(
        config["model"]
    )
    if config['model'].get('compile'):
        model = torch.compile(model)
    # load data
    # datasets = ds.load_dataset(config["dataset"], cache_dir=cache_dir)
    dataset_class = getattr(data, config["data"].pop("dataset_class"))
    train_dataset = dataset_class(**config['data'])
    val_dataset = dataset_class(**config['data'])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=config["training"]["num_workers"]
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=config["training"]["num_workers"],
    )
    
    model_tag = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print("Configuring Trainer ...")
    save_dir = os.path.join(config["training"].get("save_dir", "./checkpoints"))
    logger = []
    if config["training"].get("log_wandb"):
        logger .append( WandbLogger(
            name=model_tag,
            project=config["training"]["wandb_project"],
            save_dir=save_dir,
            group=config['training']['wandb_group']
        ))
    else: 
        logger.append(
            CSVLogger(save_dir=save_dir)
        )
    
    callback_list = []
    if config['training'].get("callbacks"):
        for callback_config in config["training"]["callbacks"]:
            callback = getattr(callbacks, callback_config.pop("callback"))
            callback_list.append(callback(**callback_config))

    trainer = Trainer(
        accelerator = config["training"].get("accelerator", "auto"),
        strategy = config["training"].get("strategy", "auto"),
        devices = config["training"].get("devices", "auto"),
        num_nodes = config["training"].get("num_nodes", 1),
        precision = config["training"].get("precision", 32),
        fast_dev_run = fast_dev_run,
        max_epochs = config["training"]["num_epochs"],
        default_root_dir = save_dir,
        logger=logger,
        callbacks=callback_list
    )

    print("Begin training...")
    trainer.fit(model, [train_dataloader], [val_dataloader])
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, model_tag + ".pt"))


if __name__=="__main__":
    train()