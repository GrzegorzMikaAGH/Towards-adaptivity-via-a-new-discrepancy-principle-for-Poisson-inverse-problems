import os

methods = ['landweber', 'tikhonov', 'tsvd']
files_to_run = os.listdir('.')

files_to_run = [f for f in files_to_run if any(m in f for m in methods) and '.py' in f]
files_to_run.sort()

with open('run_simulations.sh', 'w+') as f:
    for file in files_to_run:
        f.writelines('python3 ' + file + '\n')