# cs224n
CS224n final project


INSTRUCTIONS FOR RUNNING MAIN.py

############ SINGLE RUN OVER VALIDATION OR TEST SET #################
python code/main.py run [--num_train] [--dev or --test] [--num_dev or --num_test] [--lr] [--dropout_keep] [--reg_lambda]

############# HYPERPARAMETER VALIDATION ##########
python code/main.py validation [--num_train] [--num_dev]

EXAMPLE:
python code/main.py run --num_train=100000 --dev --num_dev --lr=0.001 --reg_lambda=0.01

(default values are used if not flag is not defined. See bottom of code/main.py)