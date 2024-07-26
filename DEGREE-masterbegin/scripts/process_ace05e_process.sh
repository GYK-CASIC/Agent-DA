export DYGIEFORMAT_PATH="./processed_data/ace05e_dygieppformat"
export OUTPUT_PATH="./processed_data/ace05e_bart"
export OUTPUT_PATH_ACEep="./processed_data/ace05ep_bart"
# bash scripts/process_ace05e_process.sh

python preprocessing/process_databyLLM.py

# python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/data.json -o $OUTPUT_PATH/main_result/AgentDA.json #-b /data01/zhanghang/txm/DEGREE-master/facebook/bartlarge

python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/data.json -o $OUTPUT_PATH_ACEep/1-shot.json -b /data01/zhanghang/txm/DEGREE-master/facebook/bartlarge


