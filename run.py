#%%
import subprocess
import tarfile
import os
import shutil
import random

# Function to execute the training command
def run_command(config):
    os.makedirs(config['output_dir'], exist_ok=True)
    command = [
        'nerfbaselines', 'train',
        '--method', config['method_name'],
        '--data', config['data_dir'],
        '--vis', 'none',
        '--backend', 'python',
        '--num-iterations', config['iterations'][-1],
        '--eval-few-iters', config['iterations'][-1],
        '--eval-all-iters', ','.join(config['iterations']),
        '--output', config['output_dir']
    ]
    try:
        print(command)
        subprocess.run(command, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")

# Function to extract specific files from tar.gz archives and organize them
def organize_outputs(config):
    for iteration in config['iterations']:
        tar_path = os.path.join(config['output_dir'], f'predictions-{iteration}.tar.gz')
        if os.path.exists(tar_path):
            extraction_path = os.path.join(
                '/home/ccl/Code/nr-nqa-dataset/outputs',
                f"{config['dataset']}/{config['scene']}/{config['method_name']}-{iteration}/"
            )
            os.makedirs(extraction_path, exist_ok=True)
            
            with tarfile.open(tar_path, "r:gz") as tar:
                def is_extractable(member):
                    return member.name.startswith('color/') or member.name == 'info.json'
                
                members_to_extract = [m for m in tar.getmembers() if is_extractable(m)]
                tar.extractall(path=extraction_path, members=members_to_extract)
                
            print(f"Extracted files for iteration {iteration} to {extraction_path}")

# Function to clean up the data directory
def clean_data_dir(data_dir):
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        else:  # For directories
            shutil.rmtree(item_path)
    print(f"Cleaned up data directory: {data_dir}")

# Main execution
if __name__ == "__main__":
    methods = ['nerfacto','instant-ngp']
    datasets = ['scannerf']#'mipnerf360', 'nerfstudio']
    dataset_scenes = {
        'mipnerf360': ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill'],
        #'scannerf': ["airplane1", "airplane2", "brontosaurus", "bulldozer1", "bulldozer2", "cheetah", "dump_truck1", "dump_truck2", "elephant", "excavator", "forklift", "giraffe"],# "helicopter1", "helicopter2", "lego", "lion", "plant1", "plant2", "plant3", "plant4", "plant5", "plant6", "plant7", "plant8", "plant9", "roadroller", "shark", "spinosaurus", "stegosaurus", "tiger", "tractor", "trex", "triceratops", "truck", "zebra"],
        'scannerf': ["helicopter1", "helicopter2", "lego", "lion", "plant1", "plant2", "plant3", "plant4", "plant5", "plant6", "plant7", "plant8", "plant9", "roadroller", "shark"],# "spinosaurus", "stegosaurus", "tiger", "tractor", "trex", "triceratops", "truck", "zebra"],
        'nerfstudio': ['aspen', 'dozer', 'egypt', 'floating-tree', 'giannini-hall', 'kitchen', 'person', 'plane', 'sculpture', 'stump']
    }
    iterations = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    for method in methods:
        for dataset in datasets:
            scenes = dataset_scenes[dataset]
            for scene in scenes:
                sampled_iterations = random.sample(iterations, 4)
                sampled_iterations.sort()
                sampled_iterations = [str(i) for i in sampled_iterations] + ['35000']
                config = {
                    'output_dir': '/home/ccl/Code/nr-nqa-dataset/outputs/tmp',
                    'data_dir': f'/home/ccl/Datasets/NeRF/{dataset}/{scene}',
                    'dataset': dataset,
                    'scene': scene,
                    'method_name': method,
                    'iterations': sampled_iterations,
                }
                run_command(config)
                organize_outputs(config)
                clean_data_dir(config['output_dir'])

# %%
