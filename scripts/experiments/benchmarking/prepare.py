import random
from multiprocessing import Pool
from typing import Callable, List, Optional, Tuple

from common import exclude_datasets, num_processors

from saga import TaskGraph
from saga.schedulers.data import Dataset, ProblemInstance
from saga.schedulers.data.random import (
    gen_in_trees,
    gen_out_trees,
    gen_parallel_chains,
    gen_random_networks,
)
from saga.schedulers.data.riotbench import (
    get_etl_task_graphs,
    get_fog_networks,
    get_predict_task_graphs,
    get_stats_task_graphs,
    get_train_task_graphs,
)
from saga.schedulers.data.wfcommons import get_networks, get_workflows


def _uses_fixed_configuration(**parameters: Optional[int]) -> bool:
    """Validate an optional all-or-none fixed graph configuration."""
    specified = [value is not None for value in parameters.values()]
    if any(specified) and not all(specified):
        missing = [name for name, value in parameters.items() if value is None]
        raise ValueError(
            "Fixed graph configuration is incomplete; missing: "
            + ", ".join(missing)
        )
    if all(specified):
        invalid = [
            name
            for name, value in parameters.items()
            if value is not None and value <= 0
        ]
        if invalid:
            raise ValueError(
                "Fixed graph configuration values must be positive: "
                + ", ".join(invalid)
            )
        return True
    return False


def in_trees_dataset(
    ccr: Optional[float] = None,
    level: Optional[int] = None,
    branch: Optional[int] = None,
    nodes: Optional[int] = None,
    num_instances: Optional[int] = None,
    overwrite: bool = False,
) -> Dataset:
    """Generate the in_trees dataset.

    Args:
        ccr: The communication to computation ratio.
        overwrite: Whether to overwrite existing instances.
        level: The levels of the in-trees.
        branch: The branching factors of the in-trees.
        nodes: The number of nodes in the networks.
        num_instances: Number of instances. Defaults to 100 in random mode and
            20 in fixed-configuration mode.

    Returns:
        Dataset: The generated dataset.
    """
    fixed_configuration = _uses_fixed_configuration(
        level=level,
        branch=branch,
        nodes=nodes,
    )
    if num_instances is None:
        num_instances = 20 if fixed_configuration else 100
    if num_instances <= 0:
        raise ValueError("num_instances must be positive")
    if ccr is not None and ccr <= 0:
        raise ValueError("ccr must be positive")

    if fixed_configuration:
        assert level is not None and branch is not None and nodes is not None
        dataset_name = f"in_trees_l{level}_b{branch}_n{nodes}"
    else:
        dataset_name = f"in_trees_ccr{ccr}" if ccr is not None else "in_trees"
    if fixed_configuration and ccr is not None:
        dataset_name += f"_ccr{ccr}"

    dataset = Dataset(name=dataset_name)
    existing_instances = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        instance_name = f"{dataset_name}_{i}"
        if instance_name in existing_instances:
            continue

        if fixed_configuration:
            processor_count = nodes
            num_levels = level
            branching_factor = branch
        else:
            processor_count = random.randint(3, 5)
            num_levels = random.randint(2, 4)
            branching_factor = random.randint(2, 3)

        network = gen_random_networks(num=1, num_nodes=processor_count)[0]
        task_graph = gen_in_trees(
            num=1,
            num_levels=num_levels,
            branching_factor=branching_factor,
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(
            ProblemInstance(
                name=instance_name,
                network=network,
                task_graph=task_graph,
            )
        )
    return dataset


def out_trees_dataset(
    ccr: Optional[float] = None,
    level: Optional[int] = None,
    branch: Optional[int] = None,
    nodes: Optional[int] = None,
    num_instances: Optional[int] = None,
    overwrite: bool = False,
) -> Dataset:
    """Generate the out_trees dataset.

    Args:
        ccr: The communication to computation ratio.
        level: The levels of the out-trees.
        branch: The branching factors of the out-trees.
        nodes: The number of nodes in the networks.
        num_instances: Number of instances. Defaults to 100 in random mode and
            20 in fixed-configuration mode.
        overwrite: Whether to overwrite existing instances.

    Returns:
        Dataset: The generated dataset.
    """
    fixed_configuration = _uses_fixed_configuration(
        level=level,
        branch=branch,
        nodes=nodes,
    )
    if num_instances is None:
        num_instances = 20 if fixed_configuration else 100
    if num_instances <= 0:
        raise ValueError("num_instances must be positive")
    if ccr is not None and ccr <= 0:
        raise ValueError("ccr must be positive")

    if fixed_configuration:
        assert level is not None and branch is not None and nodes is not None
        dataset_name = f"out_trees_l{level}_b{branch}_n{nodes}"
    else:
        dataset_name = f"out_trees_ccr{ccr}" if ccr is not None else "out_trees"
    if fixed_configuration and ccr is not None:
        dataset_name += f"_ccr{ccr}"

    dataset = Dataset(name=dataset_name)
    existing_instances = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        instance_name = f"{dataset_name}_{i}"
        if instance_name in existing_instances:
            continue

        if fixed_configuration:
            processor_count = nodes
            num_levels = level
            branching_factor = branch
        else:
            processor_count = random.randint(3, 5)
            num_levels = random.randint(2, 4)
            branching_factor = random.randint(2, 3)

        network = gen_random_networks(num=1, num_nodes=processor_count)[0]
        task_graph = gen_out_trees(
            num=1,
            num_levels=num_levels,
            branching_factor=branching_factor,
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(
            ProblemInstance(name=instance_name, network=network, task_graph=task_graph)
        )
    return dataset


def chains_dataset(
    ccr: Optional[float] = None,
    num_chains: Optional[int] = None,
    chain_length: Optional[int] = None,
    nodes: Optional[int] = None,
    num_instances: Optional[int] = None,
    overwrite: bool = False,
) -> Dataset:
    """Generate the chains dataset.

    Args:
        ccr: The communication to computation ratio.
        num_chains: Number of parallel chains.
        chain_length: Number of tasks in each parallel chain.
        nodes: The number of nodes in the networks.
        num_instances: Number of instances. Defaults to 100 in random mode and
            20 in fixed-configuration mode.
        overwrite: Whether to overwrite existing instances.

    Returns:
        Dataset: The generated dataset.
    """
    fixed_configuration = _uses_fixed_configuration(
        num_chains=num_chains,
        chain_length=chain_length,
        nodes=nodes,
    )
    if num_instances is None:
        num_instances = 20 if fixed_configuration else 100
    if num_instances <= 0:
        raise ValueError("num_instances must be positive")
    if ccr is not None and ccr <= 0:
        raise ValueError("ccr must be positive")

    if fixed_configuration:
        assert (
            num_chains is not None
            and chain_length is not None
            and nodes is not None
        )
        dataset_name = f"chains_c{num_chains}_l{chain_length}_n{nodes}"
    else:
        dataset_name = f"chains_ccr{ccr}" if ccr is not None else "chains"
    if fixed_configuration and ccr is not None:
        dataset_name += f"_ccr{ccr}"

    dataset = Dataset(name=dataset_name)
    existing_instances = set(dataset.instances) if not overwrite else set()
    for i in range(num_instances):
        instance_name = f"{dataset_name}_{i}"
        if instance_name in existing_instances:
            continue

        if fixed_configuration:
            processor_count = nodes
            current_num_chains = num_chains
            current_chain_length = chain_length
        else:
            processor_count = random.randint(3, 5)
            current_num_chains = random.randint(2, 5)
            current_chain_length = random.randint(2, 5)

        network = gen_random_networks(num=1, num_nodes=processor_count)[0]
        task_graph = gen_parallel_chains(
            num=1,
            num_chains=current_num_chains,
            chain_length=current_chain_length,
        )[0]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(
            ProblemInstance(name=instance_name, network=network, task_graph=task_graph)
        )
    return dataset


def wfcommons_dataset(
    recipe_name: str,
    ccr: float = 1.0,
    num_instances: int = 100,
    dataset_name: Optional[str] = None,
    overwrite: bool = False,
) -> Dataset:
    """Generate the wfcommons dataset.

    Args:
        recipe_name: The name of the recipe to generate the dataset for.
        ccr: The communication to computation ratio.
        num_instances: The number of instances to generate.
        dataset_name: The name of the dataset to generate. If None, the recipe
            name is used.
        overwrite: Whether to overwrite existing instances.
    """
    dataset_name = dataset_name or recipe_name
    dataset = Dataset(name=dataset_name)
    existing_instances = set(dataset.instances) if not overwrite else set()
    instance_names = {f"{dataset_name}_{i}" for i in range(num_instances)}
    new_instances = instance_names - existing_instances
    if not new_instances:
        return dataset
    networks = get_networks(num=len(new_instances), cloud_name="chameleon")
    workflows = get_workflows(num=len(new_instances), recipe_name=recipe_name)
    for i, instance_name in enumerate(new_instances):
        network = networks[i]
        task_graph = workflows[i]
        if ccr is not None:
            network = network.scale_to_ccr(task_graph, ccr)
        dataset.save_instance(
            ProblemInstance(name=instance_name, network=network, task_graph=task_graph)
        )
    return dataset


def riotbench_dataset(
    get_task_graphs: Callable[[int], List[TaskGraph]],
    name: str,
    num_instances: int = 100,
    overwrite: bool = False,
) -> Dataset:
    """Generate a riotbench dataset.

    Args:
        get_task_graphs: A function that returns a list of task graphs.
        name: The name of the dataset.
        num_instances: The number of instances to generate.
        overwrite: Whether to overwrite existing instances.

    Returns:
        Dataset: The generated dataset.
    """
    dataset = Dataset(name=name)
    existing_instances = set(dataset.instances) if not overwrite else set()
    isntance_names = {f"{name}_{i}" for i in range(num_instances)}
    new_instances = isntance_names - existing_instances
    networks = get_fog_networks(num=len(new_instances))
    task_graphs = get_task_graphs(len(new_instances))
    for i, instance_name in enumerate(new_instances):
        dataset.save_instance(
            ProblemInstance(
                name=instance_name, network=networks[i], task_graph=task_graphs[i]
            )
        )
    return dataset


def _prepare_dataset_task(task: Tuple[str, dict]) -> Tuple[str, int]:
    """Worker function to prepare a single dataset.

    Args:
        task: Tuple of (task_type, kwargs) where task_type identifies the dataset
              and kwargs are the arguments to pass to the dataset function.

    Returns:
        Tuple of (dataset_name, dataset_size).
    """
    task_type, kwargs = task

    if task_type == "in_trees":
        ds = in_trees_dataset(**kwargs)
    elif task_type == "out_trees":
        ds = out_trees_dataset(**kwargs)
    elif task_type == "chains":
        ds = chains_dataset(**kwargs)
    elif task_type == "riotbench_etl":
        ds = riotbench_dataset(get_etl_task_graphs, name="etl", **kwargs)
    elif task_type == "riotbench_predict":
        ds = riotbench_dataset(get_predict_task_graphs, name="predict", **kwargs)
    elif task_type == "riotbench_stats":
        ds = riotbench_dataset(get_stats_task_graphs, name="stats", **kwargs)
    elif task_type == "riotbench_train":
        ds = riotbench_dataset(get_train_task_graphs, name="train", **kwargs)
    elif task_type.startswith("wfcommons_"):
        recipe_name = task_type.replace("wfcommons_", "")
        ds = wfcommons_dataset(recipe_name, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    print(f"Generated dataset {ds.name} with {ds.size} instances.")
    return ds.name, ds.size


def prepare_datasets(overwrite: bool = False, num_workers: int = num_processors):
    """Generate the datasets using multiprocessing.

    Args:
        overwrite: Whether to overwrite existing instances.
        num_workers: Number of worker processes.
    """
    tasks: List[Tuple[str, dict]] = []

    # Random Graphs
    # DAG size scaling dataset, processor number is fixed to 8 and fixed CCR 1
    tasks.append(("in_trees", {"overwrite": overwrite, "level":6, "branch":2, "nodes": 8, "ccr": 1})) # 128 tasks
    tasks.append(("in_trees", {"overwrite": overwrite, "level":8, "branch":2, "nodes": 8, "ccr": 1})) # 512 tasks
    # tasks.append(("in_trees", {"overwrite": overwrite, "level":9, "branch":2, "nodes": 8, "ccr": 1})) # 1024 tasks
    tasks.append(("in_trees", {"overwrite": overwrite, "level":10, "branch":2, "nodes": 8, "ccr": 1})) # 2048 tasks
    # tasks.append(("in_trees", {"overwrite": overwrite, "level":11, "branch":2, "nodes": 8, "ccr": 1})) # 4096 tasks
    tasks.append(("in_trees", {"overwrite": overwrite, "level":12, "branch":2, "nodes": 8, "ccr": 1})) # 8192 tasks
    # tasks.append(("in_trees", {"overwrite": overwrite, "level":13, "branch":2, "nodes": 8, "ccr": 1})) # 16384 tasks
    
    tasks.append(("out_trees", {"overwrite": overwrite, "level":6, "branch":2, "nodes": 8, "ccr": 1})) # 128 tasks
    tasks.append(("out_trees", {"overwrite": overwrite, "level":8, "branch":2, "nodes": 8, "ccr": 1})) # 512 tasks
    tasks.append(("out_trees", {"overwrite": overwrite, "level":10, "branch":2, "nodes": 8, "ccr": 1})) # 2048 tasks
    tasks.append(("out_trees", {"overwrite": overwrite, "level":12, "branch":2, "nodes": 8, "ccr": 1})) # 8192 tasks
    
    tasks.append(("chains", {"overwrite": overwrite, "num_chains": 8, "chain_length": 16, "nodes": 8, "ccr": 1})) # 130 tasks
    tasks.append(("chains", {"overwrite": overwrite, "num_chains": 8, "chain_length": 64, "nodes": 8, "ccr": 1})) # 514 tasks
    tasks.append(("chains", {"overwrite": overwrite, "num_chains": 8, "chain_length": 256, "nodes": 8, "ccr": 1})) # 2050 tasks
    tasks.append(("chains", {"overwrite": overwrite, "num_chains": 8, "chain_length": 1024, "nodes": 8, "ccr": 1})) # 8194 tasks

    # Riotbench
    tasks.append(("riotbench_etl", {"overwrite": overwrite}))
    tasks.append(("riotbench_predict", {"overwrite": overwrite}))
    tasks.append(("riotbench_stats", {"overwrite": overwrite}))
    tasks.append(("riotbench_train", {"overwrite": overwrite}))

    # Wfcommons
    for recipe_name in [
        "epigenomics",
        "montage",
        "cycles",
        "seismology",
        "soykb",
        "srasearch",
        "genome",
        "blast",
        "bwa",
    ]:
        tasks.append((f"wfcommons_{recipe_name}", {"overwrite": overwrite}))

    # remove any excluded datasets
    # tasks = [task for task in tasks if task[0] not in exclude_datasets]
    tasks_not_excluded = []
    for dataset_name, kwargs in tasks:
        if any(
            excluded_dataset in dataset_name for excluded_dataset in exclude_datasets
        ):
            print(f"Skipping excluded dataset {dataset_name}")
            continue
        tasks_not_excluded.append((dataset_name, kwargs))

    with Pool(processes=num_workers) as pool:
        results = pool.map(_prepare_dataset_task, tasks_not_excluded)

    print(f"\nPrepared {len(results)} datasets.")


def main():
    prepare_datasets(overwrite=True)


if __name__ == "__main__":
    main()
