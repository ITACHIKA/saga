import logging
import os
import pathlib
from typing import Set

from saga.schedulers import (
    BILScheduler,
    CpopScheduler,
    DuplexScheduler,
    ETFScheduler,
    FastestNodeScheduler,
    FCPScheduler,
    FLBScheduler,
    GDLScheduler,
    HeftScheduler,
    MaxMinScheduler,
    MCTScheduler,
    METScheduler,
    MinMinScheduler,
    OLBScheduler,
    WBAScheduler,
)

from sagaWrapperPython import CppHeftScheduler

logging.basicConfig(level=logging.ERROR)

thisdir = pathlib.Path(__file__).parent.resolve()

# Kernel order: average computation, average inverse communication speed,
# average communication, upward rank, parent-max reduction, EFT computation,
# and argmin selection.
cpp_heft_kernel_repetitions = [
    1000,
    10000,
    1000,
    100,
    1000,
    100,
    1000,
]

cpp_heft_all_cpu_backends = ["cpu"] * 7
cpp_heft_most_simd_backends = [
    "simd",
    "simd",
    "simd",
    "cpu",
    "simd",
    "cpu",
    "simd",
]

saga_schedulers = {
    # Schedulers included in benchmarking results for the paper
    # "Comparing Task Graph Scheduling Algorithms: An Adversarial Approach"
    # https://arxiv.org/abs/2403.07120
    # "BIL": BILScheduler(),
    # "CPoP": CpopScheduler(),
    # "Duplex": DuplexScheduler(),
    # "ETF": ETFScheduler(),
    # "FCP": FCPScheduler(),
    # "FLB": FLBScheduler(),
    # "FastestNode": FastestNodeScheduler(),
    # "GDL": GDLScheduler(),
    "HEFT": HeftScheduler(),
    # "MCT": MCTScheduler(),
    # "MET": METScheduler(),
    # "MaxMin": MaxMinScheduler(),
    # "MinMin": MinMinScheduler(),
    # "OLB": OLBScheduler(),
    # "WBA": WBAScheduler(),
    # extra scheduler for CppHEFT using different configurations
    "CppHEFTallCPU_Baseline": CppHeftScheduler(
        backend_configuration=cpp_heft_all_cpu_backends,
        performance_mode="baseline",
        kernel_repetitions=cpp_heft_kernel_repetitions,
    ),
    "CppHEFTallCPU_Profile": CppHeftScheduler(
        backend_configuration=cpp_heft_all_cpu_backends,
        performance_mode="kernel_profile",
        kernel_repetitions=cpp_heft_kernel_repetitions,
    ),
    "CppHEFTmostSIMD_Baseline": CppHeftScheduler(
        backend_configuration=cpp_heft_most_simd_backends,
        performance_mode="baseline",
        kernel_repetitions=cpp_heft_kernel_repetitions,
    ),
    "CppHEFTmostSIMD_Profile": CppHeftScheduler(
        backend_configuration=cpp_heft_most_simd_backends,
        performance_mode="kernel_profile",
        kernel_repetitions=cpp_heft_kernel_repetitions,
    ),
}

exclude_datasets: Set[str] = {
    "bwa",
    "cycles",
    "epigenomics",
    "genome",
    "montage",
    "seismology",
    "soykb",
    "srasearch",
    "blast",
    # exclude the following for testing purpose since cppHeft and Heft produce identical result
    # "etl",
    # "in_trees",
    # "out_trees",
    # "predict",
    # "stats"
}

datadir = thisdir.joinpath("data", "benchmarking")
resultsdir = thisdir.joinpath("results", "benchmarking")
outputdir = thisdir.joinpath("output", "benchmarking")

num_processors = max(1, (os.cpu_count() or 1) - 3)

os.environ["SAGA_DATA_DIR"] = str(thisdir.joinpath("data", "benchmarking"))
