# python RLHF/select_LLM_SLM.py
import json

# Read index values from requirement.txt file
with open('RLHF/accuracy.txt', 'r') as file:
    output_indices = [int(line.strip()) for line in file.readlines()]

# Get indices with values 0 and 1
indices_to_output_LLM = [i for i, value in enumerate(output_indices) if value == 0]
indices_to_output_SLM = [i for i, value in enumerate(output_indices) if value == 1]

# Read JSON files
with open('Collaborative_Filtering/diff_LLM.json', 'r') as file:
    data_LLM = json.load(file)
with open('Collaborative_Filtering/diff_SLM.json', 'r') as file:
    data_SLM = json.load(file)

# Selectively output data
output_data_LLM = [data_LLM[i] for i in indices_to_output_LLM]
output_data_SLM = [data_SLM[i] for i in indices_to_output_SLM]

with open('Collaborative_Filtering/high_confidence_SLM.json', 'r') as file:
    high_confidence_SLM = json.load(file)

with open('Collaborative_Filtering/matched_entries.json', 'r') as file:
    matched_entries = json.load(file)

with open('processed_data/data_LLM/1-shot.json', 'r') as file:
    raw_data = json.load(file)

merged_data = output_data_LLM + output_data_SLM
augmented_data = output_data_LLM + output_data_SLM + high_confidence_SLM + matched_entries + raw_data

# Save the output data to new JSON files
with open('RLHF/selected_data_LLM.json', 'w') as outfile:
    json.dump(output_data_LLM, outfile, indent=4)
with open('RLHF/selected_data_SLM.json', 'w') as outfile:
    json.dump(output_data_SLM, outfile, indent=4)

with open('RLHF/merged_data.json', 'w') as outfile:
    json.dump(merged_data, outfile, indent=4)
with open('augmented_sample.json', 'w') as outfile:
    json.dump(augmented_data, outfile, indent=4)

print("Data processing completed.")
