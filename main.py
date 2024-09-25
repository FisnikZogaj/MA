import subprocess
import time
import multiprocessing as mp
from datetime import datetime
import os
from config import Scenarios

fail_count = 0
base = r"C:\Users\zogaj\PycharmProjects\MA\ExperimentLogs"  # my machine
# base = r"/home/zogaj/MA/ExperimentLogs"  # Linux Server

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

    python = r'C:\Users\zogaj\PycharmProjects\MA\venv\Scripts\python.exe' # my machine
    #python = r'/home/zogaj/MA/genv/bin/python'  # Linux server
    config_str = str(config)  # Convert dictionary to string for argparse
    command = [
        python, 'train.py',  # The script to run on the version of python specified
        '--config', config_str,
        '--architecture', architecture,
        '--seed', str(seed),
        '--timestamp', timestamp,
    ]
    process = subprocess.Popen(command)
    process.wait()  # Wait for the process to complete!
    return process.returncode

# Function to run job and handle exceptions
def run_job_safe(args, counter, lock, total_num_of_jobs):
    global fail_count
    try:
        result = run_job(*args)
        with lock:
            counter.value += 1
            print(f"{counter.value}/{total_num_of_jobs} done!")
        return result
    except Exception as e:
        print(f"Job with args {args} generated an exception: {e}")
        fail_count += 1
        return None

if __name__ == '__main__':
    start_time = time.time()
    # Note: range determines the number of Monte-Carlo runs
    seeds = list(range(1, 3))
    arguments = Scenarios()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S").translate(str.maketrans({" ": "-", ":": "-"}))
    # List of arguments for the jobs; Permutation of all settings
    job_args = [(c, a, s, ts) for c in arguments
                for a in ["GCN", "SAGE", "GAT", "XGBoost"]
                for s in seeds]

    total_num_of_jobs = len(job_args)

    # Before jobs are executed:
    # ---- Make directories for saving Train results ----

    subdir_path = os.path.join(base, ts)
    os.makedirs(subdir_path, exist_ok=True)

    for arch in ["GCN", "SAGE", "GAT", "XGBoost"]:
        for graphtype in arguments.list_of_scenarios:
            arch_path = os.path.join(subdir_path, arch)
            graphtype_path = os.path.join(arch_path, graphtype)
            os.makedirs(graphtype_path, exist_ok=True)

    # ---- Make directories for saving Graph Characteristics ---

    for graphtype in arguments.list_of_scenarios:
        # purity: gini and entropy for com/centroid
        # tec: target edge counter
        # lab_corr: 4 metrics for y~c, y~f, c~f
        gchar_path = os.path.join(subdir_path, "GraphCharacteristics")
        gtype_path = os.path.join(gchar_path, graphtype)
        os.makedirs(gtype_path, exist_ok=True)


    # ------------ Save the configs used for Graph generating and training -------------------

    config_file_path = 'config.py'
    output_config_path = os.path.join(subdir_path, 'configs.txt')

    with open(config_file_path, 'r') as input_file:
        script_content = input_file.read()

    with open(output_config_path, 'w') as output_file:
        output_file.write(script_content)


    model_file_path = 'GNN_Models.py'
    output_models_path = os.path.join(subdir_path, 'models.txt')

    with open(model_file_path, 'r') as input_file:
        script_content = input_file.read()

    with open(output_models_path, 'w') as output_file:
        output_file.write(script_content)

    # --------------- Start Multiprocess training -------------------

    manager = mp.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    max_concurrent_jobs = 2

    with mp.Pool(processes=max_concurrent_jobs) as pool:
        results = [pool.apply_async(run_job_safe, args=(arg, counter, lock, total_num_of_jobs)) for arg in job_args]
        for result in results:
            result.get()  #  Ensure each result is processed

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
