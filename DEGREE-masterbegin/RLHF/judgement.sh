#!/bin/bash

# Combine annotations of large and small models
python RLHF/process.py

# Adjudicator judgment
python RLHF/inference_reward_model.py

# Output the final filtered data
python RLHF/select_LLM_SLM.py

# bash RLHF/judgement.sh
