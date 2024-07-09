import subprocess
import time
import multiprocessing as mp
from config import Scenarios

fail_count = 0
def run_job(config: dict, architecture: str, seed: int):
    config_str = str(config)  # Convert dictionary to string
    command = [
        'python', 'train.py',  # The script to run
        '--config', config_str,
        '--architecture', architecture,
        '--seed', str(seed)
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


start_time = time.time()

# range determines the number of Monte-Carlo runs
seeds = list(range(1, 10))
arguments = Scenarios()

# List of arguments for the jobs
job_args = [(c, a, s) for c in arguments
            for a in ["GCN", "SAGE", "GAT"]
            for s in seeds]

# Before jobs are executed, create directories to store the results
# With timestamp?
#/GAT/perfect/res1.pkl
#...
##/GAT/perfect/res100.pkl

max_concurrent_jobs = 4
with mp.Pool(processes=max_concurrent_jobs) as pool:
    results = pool.map(run_job_safe, job_args)

print(f"{(1-(fail_count/len(job_args))) * 100} % of training jobs have completed !")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
