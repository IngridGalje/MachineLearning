from pathlib import Path
from typing import Dict

import ray
import torch
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics, rnn_models
from mltrainer.preprocessors import PaddedPreprocessor
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

##SAMPLE_INT = tune.search.sample.Integer    ##dit is niet nodig bij een grid search
##SAMPLE_FLOAT = tune.search.sample.Float    ##dit is niet nodig bij een grid search

#NUM_SAMPLES = 10  # Dit is niet nodig voor grid search, maar kan blijven staan
#MAX_EPOCHS = 2


def train(config: Dict):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """
    try:
        from mads_datasets import DatasetFactoryProvider, DatasetType

        data_dir = config["data_dir"]
        gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
        preprocessor = PaddedPreprocessor()

        with FileLock(data_dir / ".lock"):
            # we lock the datadir to avoid parallel instances trying to
            # access the datadir
            streamers = gesturesdatasetfactory.create_datastreamer(
                batchsize=32, preprocessor=preprocessor
            )
            train = streamers["train"]
            valid = streamers["valid"]

        # we set up the metric
        # and create the model with the config
        accuracy = metrics.Accuracy()
        model = rnn_models.GRUmodel(config)

        trainersettings = TrainerSettings(
            epochs=5,                   #geen max epochs meer om eventueel te kunnen stoppen
            metrics=[accuracy],
            logdir=Path("."),
            train_steps=len(train),  # type: ignore
            valid_steps=len(valid),  # type: ignore
            reporttypes=[ReportTypes.RAY],
            scheduler_kwargs={"factor": 0.5, "patience": 5},
            earlystop_kwargs=None,
        )

        # because we set reporttypes=[ReportTypes.RAY]
        # the trainloop wont try to report back to tensorboard,
        # but will report back with ray
        # this way, ray will know whats going on,
        # and can start/pause/stop a loop.
        # This is why we set earlystop_kwargs=None, because we
        # are handing over this control to ray.

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = "cpu"  # type: ignore
        logger.info(f"Using {device}")
        if device != "cpu":
            logger.warning(
                f"using acceleration with {device}." "Check if it actually speeds up!"
            )

        optimizer = torch.optim.Adam if config["optimizer"] == "Adam" else torch.optim.SGD
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

        trainer = Trainer(
            model=model,
            settings=trainersettings,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,  # type: ignore
            traindataloader=train.stream(),
            validdataloader=valid.stream(),
            scheduler=scheduler,   #omzetten voor grid
            device=device,
        )

        logger.info("Starting training")
        trainer.loop()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    ray.init()

    data_dir = Path("data/raw/gestures/gestures-dataset").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    tune_dir = Path("notebooks/4_tuning_networks/models/ray").resolve()
    #search = HyperOptSearch()
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=1,
        reduction_factor=3,
        max_t=5,
    )

    config = {
        "input_size": 3,
        "output_size": 20,
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "hidden_size": tune.grid_search([32, 64, 128, 256]),  #qrandint verandert in grid_search
        "dropout": tune.grid_search([0.1, 0.2, 0.3]),      #uniform verandert in grid_search
        "num_layers": tune.grid_search([2, 3, 4, 5]),         #randint verandert in grid_search 
        "optimizer": tune.choice(["SGD", "Adam"]),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=str(config["tune_dir"]),
       # num_samples=NUM_SAMPLES,
        #search_alg=search,
        scheduler=scheduler,
        verbose=1,
    )

    ray.shutdown()
