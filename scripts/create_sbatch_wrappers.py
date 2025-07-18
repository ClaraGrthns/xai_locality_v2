#!/usr/bin/env python3
import os
import re
import glob
from pathlib import Path

def determine_resources(script_content):
    """Determine resource requirements based on model type in the script."""
    
    # Default values
    resources = {
        "partition": "day",
        "cpus_per_task": "8",
        "mem_per_cpu": "8G",
        "gres": "gpu:0",
        "time": "3:00:00"
    }
    if "force_training" in script_content:
        resources["gres"] = "gpu:1"
        resources["time"] = "2:00:00"
        for model in ["TabNet", "FTTransformer", "TabTransformer"]:
            if model in script_content:
                resources["time"] = "5:00:00" if "higgs" in script_content else "4:00:00"
                break
    
    return resources

def extract_job_name(sh_file_path):
    base_name = os.path.basename(sh_file_path)
    job_name = os.path.splitext(base_name)[0]
    path_parts = sh_file_path.split(os.sep)

    for model in ["TabNet", "FTTransformer", "MLP", "LogReg", "LightGBM", "ResNet", "TabTransformer"]:
        if model in path_parts:
            job_name = f"{model}_{job_name}"
    return job_name
    

def create_sbatch_wrapper(sh_file_path):
    """Create an sbatch wrapper script for the given shell script."""
    
    # Skip run_all.sh files
    if os.path.basename(sh_file_path) == "run_all.sh":
        return None
    
    # Read the content of the shell script
    with open(sh_file_path, 'r') as f:
        script_content = f.read()
    
    # Generate job name
    job_name = extract_job_name(sh_file_path)
    
    # Get appropriate resources
    resources = determine_resources(script_content)
    
    # Create output directory for sbatch files
    sbatch_dir = os.path.join(os.path.dirname(sh_file_path))
    os.makedirs(sbatch_dir, exist_ok=True)
    
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(sh_file_path), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    output_path = os.path.join(sbatch_dir, f"{job_name}.sbatch")

    
    # Create sbatch content with the format specified
    sbatch_content = f"""#!/bin/bash

####
# a) Define slurm job parameters
####

#SBATCH --job-name={job_name}

#resources:
#SBATCH --cpus-per-task={resources['cpus_per_task']}
#SBATCH --partition={resources['partition']}
#SBATCH --mem-per-cpu={resources['mem_per_cpu']}
#SBATCH --gres={resources['gres']}
#SBATCH --time={resources['time']}

#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --output={log_dir}/{job_name}_%j.out

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=clara.grotehans@student.uni-tuebingen.de

echo "Job started at $(date)"

# Execute the command from the original script
{sh_file_path}

echo "-------------------------------------"
echo "Job completed at $(date)"
"""
    
    # Write the sbatch wrapper
    with open(output_path, 'w') as f:
        f.write(sbatch_content)
    
    # Make executable
    os.chmod(output_path, 0o755)
    
    return output_path

def main():
    """Create sbatch wrappers for all shell scripts in experiment_commands."""
    
    # Find the experiment_commands directory
    base_dir = Path(__file__).parent.parent  # xai_locality root
    experiment_dir = os.path.join(base_dir, 'commands_sbach_files', 'experiment_commands')
    experiment_dir = "/home/grotehans/xai_locality_v2/commands_sbach_files/experiment_commands/shap/LightGBM"
    if not os.path.exists(experiment_dir):
        print(f"Directory {experiment_dir} not found")
        return
    
    # Find all .sh files only within folders named "synthetic_data" and exclude run_all.sh files
    # Find all .sh files but exclude run_all.sh files
    all_sh_files = [
        f for f in glob.glob(os.path.join(experiment_dir, "**/*.sh"), recursive=True)
        if ( "euclidean" in f)
    ]
    # sh_files = [f for f in all_sh_files if (os.path.basename(f) != "run_all.sh")]
    # all_sh_files = glob.glob(os.path.join(experiment_dir, "**/*.sh"), recursive=True)

    sh_files = [f for f in all_sh_files if os.path.basename(f) != "run_all.sh"]
    if not sh_files:
        print(f"No individual experiment shell scripts found in {experiment_dir}")
        return
    
    print(f"Found {len(sh_files)} individual experiment shell scripts")
    
    # Create sbatch wrappers
    created_count = 0
    for sh_file in sh_files:
        sbatch_file = create_sbatch_wrapper(sh_file)
        if sbatch_file:
            created_count += 1
            print(f"Created SBATCH wrapper: {sbatch_file}")
    
    print(f"\nCreated {created_count} sbatch wrapper files successfully.")
    print("\nTo submit a job, use: sbatch path/to/file.sbatch")
    print("\nTo submit all jobs in a directory:")
    print("  cd /path/to/sbatch/dir")
    print("  for f in *.sbatch; do sbatch $f; done")

if __name__ == "__main__":
    main()
