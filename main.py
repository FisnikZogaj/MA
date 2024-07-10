import subprocess
import time
import multiprocessing as mp
from datetime import datetime
import os

from config import Scenarios

fail_count = 0
def run_job(config: dict, architecture: str, seed: int, timestamp: str):
    """
    Run an entire train script that trains a graph (specified from config) with
    a model (architecture) and a seed, for reproducibility.
    :param config:
    :param architecture:
    :param seed:
    :param timestamp:
    :return:
    """
    config_str = str(config)  # Convert dictionary to string for argparse
    python = r'C:\Users\zogaj\PycharmProjects\MA\venv\Scripts\python.exe'
    command = [
        python, 'train.py',  # The script to run on the version of python specified
        '--config', config_str,
        '--architecture', architecture,
        '--seed', str(seed),
        '--timestamp', timestamp,
    ]
    process = subprocess.Popen(command)
    process.wait()  # Wait for the process to complete
    return process.returncode


# Function to run job and handle exceptions
def run_job_safe(args):
    global fail_count
    try:
        return run_job(*args)
    except Exception as e:
        print(f"Job with args {args} generated an exception: {e}")
        fail_count += 1
        return None

if __name__ == '__main__':
    start_time = time.time()
    # Note: range determines the number of Monte-Carlo runs
    seeds = list(range(1, 10))
    arguments = Scenarios()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S").translate(str.maketrans({" ": "-", ":": "-"}))
    # List of arguments for the jobs; Permutation of all settings
    job_args = [(c, a, s, ts) for c in arguments
                for a in ["GCN", "SAGE", "GAT"]
                for s in seeds]

    # Before jobs are executed:
    # ---- Make directories for saving ----

    #ExperimentLogs/ts/GAT/perfect/res1.pkl
    #...
    #ExperimentLogs/ts/GAT/perfect/resN.pkl

    base = r"C:\Users\zogaj\PycharmProjects\MA\ExperimentLogs"
    subdir_path = os.path.join(base, ts)
    os.makedirs(subdir_path, exist_ok=True)

    for arch in ["GCN", "SAGE", "GAT"]:
        for graphtype in arguments.list_of_scenarios:
            # Make sure the list has same order as the dictionaries from Scenarios
            # from: self._elements = [self.perfect, ... , self.noise]
            arch_path = os.path.join(subdir_path, arch)
            graphtype_path = os.path.join(arch_path, graphtype)
            os.makedirs(graphtype_path, exist_ok=True)

    # ------------ Save the configs used for Graph generating and training -------------------

    input_file_path = 'config.py'
    output_file_path = os.path.join(subdir_path, 'configs.txt')

    # Copy the code from the script and write it to configs.txt.
    with open(input_file_path, 'r') as input_file:
        script_content = input_file.read()

    with open(output_file_path, 'w') as output_file:
        output_file.write(script_content)

    # --------------- Start Multiprocess training -------------------
    print("start: ")
    max_concurrent_jobs = 4
    with mp.Pool(processes=max_concurrent_jobs) as pool:
        results = pool.map(run_job_safe, job_args)

    print(f"{(1-(fail_count/len(job_args))) * 100} % of training jobs have completed !")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
