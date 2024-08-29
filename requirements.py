import future
import hyppo
import graspy
import numpy
import pandas
import torch_geometric
import torch
import networkx


def check_version(package, actual_version, expected_version):
    if actual_version == expected_version:
        print(f"{package}: True")
    else:
        print(f"{package}: False, actual version is {actual_version}")

# Define the expected versions
expected_versions = {
    "hyppo": "0.1.3",
    "future": "1.0.0",
    "graspy": "0.3.0",
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "torch_geometric": "2.6.0",
    "torch": "2.3.1+cpu",
    "networkx": "3.3"
}

# Check each package
check_version("hyppo", hyppo.__version__, expected_versions["hyppo"])
check_version("future", future.__version__, expected_versions["future"])
check_version("graspy", graspy.__version__, expected_versions["graspy"])
check_version("numpy", numpy.__version__, expected_versions["numpy"])
check_version("pandas", pandas.__version__, expected_versions["pandas"])
check_version("torch_geometric", torch_geometric.__version__, expected_versions["torch_geometric"])
check_version("torch", torch.__version__, expected_versions["torch"])
check_version("networkx", networkx.__version__, expected_versions["networkx"])
