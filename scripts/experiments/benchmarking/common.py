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
    "CppHEFTallCPU_PhaseOnly_FullStats": CppHeftScheduler(
        backend_configuration=cpp_heft_all_cpu_backends,
        measurement_region="phase_profile",
        measurement_mode="full_stats",
    ),
    "CppHEFTallCPU_PhaseOnly_TimeOnly": CppHeftScheduler(
        backend_configuration=cpp_heft_all_cpu_backends,
        measurement_region="phase_profile",
        measurement_mode="time_only",
    ),
    "CppHEFTallCPU_Kernel_FullStats": CppHeftScheduler(
        backend_configuration=cpp_heft_all_cpu_backends,
        measurement_region="kernel_profile",
        measurement_mode="full_stats",
    ),
    "CppHEFTallCPU_Kernel_TimeOnly": CppHeftScheduler(
        backend_configuration=cpp_heft_all_cpu_backends,
        measurement_region="kernel_profile",
        measurement_mode="time_only",
    ),
    "CppHEFTmostSIMD_PhaseOnly_FullStats": CppHeftScheduler(
        backend_configuration=cpp_heft_most_simd_backends,
        measurement_region="phase_profile",
        measurement_mode="full_stats",
    ),
    "CppHEFTmostSIMD_PhaseOnly_TimeOnly": CppHeftScheduler(
        backend_configuration=cpp_heft_most_simd_backends,
        measurement_region="phase_profile",
        measurement_mode="time_only",
    ),
    "CppHEFTmostSIMD_Kernel_FullStats": CppHeftScheduler(
        backend_configuration=cpp_heft_most_simd_backends,
        measurement_region="kernel_profile",
        measurement_mode="full_stats",
    ),
    "CppHEFTmostSIMD_Kernel_TimeOnly": CppHeftScheduler(
        backend_configuration=cpp_heft_most_simd_backends,
        measurement_region="kernel_profile",
        measurement_mode="time_only",
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
    # "in_trees",
    # "out_trees",
    # "chains",
    "predict",
    "stats",
    "train",
    "etl",
}

datadir = thisdir.joinpath("data", "benchmarking")
resultsdir = thisdir.joinpath("results", "benchmarking")
outputdir = thisdir.joinpath("output", "benchmarking")

num_processors = max(1, (os.cpu_count() or 1) - 3)

os.environ["SAGA_DATA_DIR"] = str(thisdir.joinpath("data", "benchmarking"))
