export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
'''
DATASET="ace05e"
# DATASET="ace05ep"

nohup bash scripts/train_degree_e2e.sh&
python degree/generate_data_degree_e2e.py -c config/config_degree_e2e_ace05e.json
python degree/train_degree_e2e1.py -c config/config_degree_e2e_ace05e.json
nohup python degree/train_degree_e2e1.py -c config/config_degree_e2e_ace05e.json >5-shotSR.txt 2>&1 &
'''
# bash scripts/train_degree_e2e.sh

# DATASET="ace05ep"
python degree/generate_data_degree_e2e.py -c config/config_degree_e2e_ace05ep_test.json
python degree/train_degree_e2e1.py -c config/config_degree_e2e_ace05ep_test.json
# nohup python degree/train_degree_e2e1.py -c config/config_degree_e2e_ace05ep.json >1-shoteda.txt 2>&1 &


# python -m debugpy --listen 1234 --wait-for-client degree/train_degree_e2e.py -c config/config_degree_e2e_ace05e.json