from datetime import datetime
import logging

from run_exprtiment import run
from exp_parameters import TASKS

logging.basicConfig(level=logging.INFO)

print("Starting...")
start_t = datetime.now()
task = TASKS['Food101']
run(task)
run(task, mode='svd')
run(task, mode='sfp')
print(f'Total time: {datetime.now() - start_t}')
