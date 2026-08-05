import logging
import pathlib
import random
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import filelock
import pandas as pd
from common import (
    datadir,
    exclude_datasets,
    num_processors,
    resultsdir,
    saga_schedulers,
)
from tqdm import tqdm

from saga import Scheduler
from saga.schedulers.data import Dataset

import pdb

# Global variables for worker processes
_worker_resultsdir: pathlib.Path
_worker_schedulers: Dict[str, Scheduler]


def _init_worker(results_dir: pathlib.Path, schedulers: Dict[str, Scheduler]):
    """Initialize worker process with shared state."""
    global _worker_resultsdir, _worker_schedulers
    _worker_resultsdir = results_dir
    _worker_schedulers = schedulers


def _evaluate_instance(args: Tuple[str, str]) -> List[Dict]:
    """Evaluate a single instance with all schedulers.

    Args:
        args: Tuple of (dataset_name, instance_name)

    Returns:
        List of result dictionaries.
    """
    dataset_name, instance_name = args
    dataset = Dataset(name=dataset_name)
    instance = dataset.get_instance(instance_name)
    savepath = _worker_resultsdir.joinpath(f"{dataset_name}.csv")
    lock_path = savepath.with_suffix(".csv.lock")

    cpp_heft_repeats = 3

    results = []
    for scheduler_name, scheduler in _worker_schedulers.items():
        schedules = [] # use a list to store the current scheduler's repeated runs
        current_results = []
        # Check if already finished (with lock to avoid race condition)
        with filelock.FileLock(lock_path):
            if savepath.exists():
                finished_df = pd.read_csv(savepath)
                finished = set(
                    zip(
                        finished_df["Dataset"],
                        finished_df["Instance"],
                        finished_df["Scheduler"],
                    )
                )
                if (dataset_name, instance_name, scheduler_name) in finished:
                    logging.info(
                        "  %s/%s/%s already finished. Skipping.",
                        dataset_name,
                        instance_name,
                        scheduler_name,
                    )
                    continue
        if("CppHEFT" in scheduler_name): # special case for CppHEFT schedulers to allow extra params
            extra_params = {}
            extra_params["dataset_name"] = dataset_name
            extra_params["instance_name"] = instance_name
            extra_params["scheduler_name"] = scheduler_name
            for i in range(1,cpp_heft_repeats+1):
                extra_params["repeats_id"] = i
                schedules.append(scheduler.schedule(
                    network=instance.network, task_graph=instance.task_graph, extra_param = extra_params
                ))
        else:
            schedules.append(scheduler.schedule(
                network=instance.network, task_graph=instance.task_graph
            ))
        # print(scheduler_name)
        # pdb.set_trace()
        # schedule = CppHeftScheduler().schedule(
        #     network=instance.network, task_graph=instance.task_graph)
        for repeat_id, schedule in enumerate(schedules, start=1):
            makespan = schedule.makespan
            result = {
                "Dataset": dataset_name,
                "Instance": instance_name,
                "Scheduler": scheduler_name,
                "Repeat": repeat_id,
                "Makespan": makespan,
            }
            results.append(result)
            current_results.append(result)

        # Write result immediately with lock
        with filelock.FileLock(lock_path):
            result_df = pd.DataFrame(current_results)
            if savepath.exists():
                result_df.to_csv(savepath, mode="a", header=False, index=False)
            else:
                result_df.to_csv(savepath, index=False)

        logging.info(
            "  %s/%s/%s: %.4f", dataset_name, instance_name, scheduler_name, makespan
        )

    return results


def evaluate_dataset(
    resultsdir: pathlib.Path,
    dataset_name: str,
    schedulers: Optional[Dict[str, Scheduler]] = None,
    overwrite: bool = False,
    num_workers: int = num_processors,
    shuffle: bool = True,
    seed: int = 42,
):
    """Evaluate a dataset using multiprocessing.

    Args:
        resultsdir: The directory to save the results.
        dataset_name: The name of the dataset.
        schedulers: The schedulers to evaluate. Defaults to None (all schedulers).
        overwrite: Whether to overwrite existing results. Defaults to False.
        num_workers: Number of worker processes.
        shuffle: Whether to shuffle instances to distribute work evenly.
        seed: Random seed for shuffling.
    """
    savepath = resultsdir.joinpath(f"{dataset_name}.csv")
    savepath.parent.mkdir(exist_ok=True, parents=True)

    if overwrite and savepath.exists():
        savepath.unlink()

    schedulers = schedulers or saga_schedulers
    dataset = Dataset(name=dataset_name)
    instance_names = dataset.instances

    logging.info("Dataset %s has %d instances.", dataset_name, len(instance_names))

    # Create list of (dataset_name, instance_name) tuples
    work_items = [(dataset_name, name) for name in instance_names]

    # Shuffle to distribute expensive instances across workers
    if shuffle:
        random.Random(seed).shuffle(work_items)

    # Run in parallel with progress bar
    # with Pool(
    #     processes= num_workers,
    #     initializer=_init_worker,
    #     initargs=(resultsdir, schedulers),
    # ) as pool:
    #     list(
    #         tqdm(
    #             pool.imap_unordered(_evaluate_instance, work_items),
    #             total=len(work_items),
    #             desc=f"Evaluating {dataset_name}",
    #             unit="instance",
    #         )
    #     )
    
    # Serial execution for debugging
    _init_worker(resultsdir, schedulers)
    list(
        tqdm(
            map(_evaluate_instance, work_items),
            total=len(work_items),
            desc=f"Evaluating {dataset_name}",
            unit="instance",
        )
    )


def main():
    datasets = [path.name for path in datadir.iterdir() if path.is_dir()]
    for dataset in datasets:
        if dataset in exclude_datasets:
            logging.info("Skipping excluded dataset %s", dataset)
            continue
        logging.info("Evaluating dataset %s", dataset)
        evaluate_dataset(resultsdir, dataset, overwrite=True)


if __name__ == "__main__":
    main()
