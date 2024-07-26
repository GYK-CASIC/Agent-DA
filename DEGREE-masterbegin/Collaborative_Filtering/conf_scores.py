import json
import os

input_data_dir = "Collaborative_Filtering"
output_dir = "Collaborative_Filtering"

# Read the data file
with open(os.path.join("output/ACE05-ep/my_output/pred.test.json"), "r") as file:
    data = json.load(file)

# Filter data, removing entries with a confidence score equal to 100, indicating the event sample does not belong to this event type
filtered_data = [entry for entry in data if entry["gold text"] != "Event trigger is <Trigger>"]

# Write the filtered results to the output file
filtered_output_path = os.path.join(output_dir, "process_test.json")
with open(filtered_output_path, "w") as file:
    json.dump(filtered_data, file, indent=4)
print(f"Filtered data written to: {filtered_output_path}")

# Read the filtered data file
with open(filtered_output_path, "r") as file:
    data = json.load(file)

with open(os.path.join("processed_data/data_LLM/1-shotchat.json"), "r") as file:
    gold_data = json.load(file)

# Filter entries with different annotations in input_text
passages = [entry["gold info"][0][0]["passage"] for entry in data if entry["gold text"] != entry["pred text"]]

# Divide data into same annotation and different annotation data
same_text_data = []
diff_text_data = []
for entry in data:
    if entry["gold info"][0][0]["passage"] in passages:
        diff_text_data.append(entry)
    else:
        same_text_data.append(entry)

# Output data with different annotations
diff_text_data_path = os.path.join(output_dir, "diff_text_data.json")
with open(diff_text_data_path, "w") as file:
    json.dump(diff_text_data, file, indent=4)
print(f"Data with different annotations written to: {diff_text_data_path}")

# Output data with the same annotations
same_text_data_path = os.path.join(output_dir, "same_text_data.json")
with open(same_text_data_path, "w") as file:
    json.dump(same_text_data, file, indent=4)
print(f"Data with the same annotations written to: {same_text_data_path}")

# Find sentence information with the same annotations
matched_entries = []
diff_entries = []
first_sentences = [entry["gold info"][0][0]["passage"] for entry in same_text_data]
diff_sentences = [entry["gold info"][0][0]["passage"] for entry in diff_text_data]
for entry in gold_data:
    for text in first_sentences:
        if entry["sentence"].replace(" ", "") == text.replace(" ", "") and entry not in matched_entries:
            matched_entries.append(entry)
    for text in diff_sentences:
        if entry["sentence"].replace(" ", "") == text.replace(" ", "") and entry not in diff_entries:
            diff_entries.append(entry)

# Output sentence information with different annotations
diff_entries_path = os.path.join(output_dir, "diff_entries.json")
with open(diff_entries_path, "w") as file:
    json.dump(diff_entries, file, indent=4)
print(f"Sentence information with different annotations written to: {diff_entries_path}")

# Output sentence information with the same annotations
matched_entries_path = os.path.join(output_dir, "matched_entries.json")
with open(matched_entries_path, "w") as file:
    json.dump(matched_entries, file, indent=4)
print(f"Sentence information with the same annotations written to: {matched_entries_path}")

# Find samples that the small model cannot identify
low_confidence_data = [entry for entry in diff_text_data if 5 <= entry["confidence scores"] <= 95]
print("Low confidence data:", low_confidence_data)

# For low confidence, convert it to both real and predicted formats for reward model processing
# Low confidence large model predictions
diff_LLM = []
second_sentences = [entry["gold info"][0][0]["passage"] for entry in low_confidence_data]
for entry in gold_data:
    for text in second_sentences:
        if entry["sentence"].replace(" ", "") == text.replace(" ", "") and entry not in diff_LLM:
            entry["sentence"] = text
            diff_LLM.append(entry)

# Low confidence small model predictions
diff_SLM = []
for entry in low_confidence_data:
    sentence = entry["gold info"][0][0]["passage"]
    pred_triggers = entry["pred triggers"]
    if pred_triggers:
        pred_info_entry = {
            "sentence": sentence,
            "events_info": [{
                "trigger_text": pred_triggers[0][0],
                "event_type": pred_triggers[0][1],
                "arguments": [
                    {"text": arg[0], "role": arg[1]} for arg in pred_triggers[1:]
                ]
            }]
        }
    else:
        pred_info_entry = {
            "sentence": sentence,
            "events_info": [{
                "trigger_text": None,
                "event_type": None,
                "arguments": []
            }]
        }
    diff_SLM.append(pred_info_entry)

# Merge low confidence data
merged_data = {}
for entry in diff_SLM:
    sentence = entry["sentence"]
    events_info = entry["events_info"]
    if sentence in merged_data:
        merged_data[sentence]["events_info"].extend(events_info)
    else:
        merged_data[sentence] = {
            "sentence": sentence,
            "events_info": events_info
        }

# Write the merged low confidence data to a new JSON file
merged_data_list = list(merged_data.values())
diff_SLM_output = os.path.join(output_dir, 'diff_SLM.json')
with open(diff_SLM_output, "w", encoding="utf-8") as f:
    json.dump(merged_data_list, f, indent=4)
print(f"Low confidence small model annotation data written to: {diff_SLM_output}")

# Different annotation small model predictions
all_diff_SLM = []
for entry in diff_text_data:
    sentence = entry["gold info"][0][0]["passage"]
    pred_triggers = entry["pred triggers"]
    if pred_triggers:
        pred_info_entry = {
            "sentence": sentence,
            "events_info": [{
                "trigger_text": pred_triggers[0][0],
                "event_type": pred_triggers[0][1],
                "arguments": [
                    {"text": arg[0], "role": arg[1]} for arg in pred_triggers[1:]
                ]
            }]
        }
    else:
        pred_info_entry = {
            "sentence": sentence,
            "events_info": [{
                "trigger_text": None,
                "event_type": None,
                "arguments": []
            }]
        }
    all_diff_SLM.append(pred_info_entry)

# Merge different annotation small model prediction data
merged_data = {}
for entry in all_diff_SLM:
    sentence = entry["sentence"]
    events_info = entry["events_info"]
    if sentence in merged_data:
        merged_data[sentence]["events_info"].extend(events_info)
    else:
        merged_data[sentence] = {
            "sentence": sentence,
            "events_info": events_info
        }

# Write the merged data to a new JSON file
merged_data_list = list(merged_data.values())
all_diff_SLM_output = os.path.join(output_dir, 'all_diff_SLM.json')
with open(all_diff_SLM_output, "w", encoding="utf-8") as f:
    json.dump(merged_data_list, f, indent=4)
# print(f"Different annotation small model annotation data written to: {all_diff_SLM_output}")

# Find different annotation small model annotation information minus low confidence small model annotation data
high_confidence_SLM = []
low_confidence_sentences = [entry["sentence"] for entry in diff_SLM]
for entry in all_diff_SLM:
    if entry["sentence"] not in low_confidence_sentences:
        high_confidence_SLM.append(entry)

# Write the high confidence small model annotation information to a new JSON file
high_confidence_SLM_output = os.path.join(output_dir, 'high_confidence_SLM.json')
with open(high_confidence_SLM_output, "w", encoding="utf-8") as f:
    json.dump(high_confidence_SLM, f, indent=4)
print(f"High confidence small model annotation information written to: {high_confidence_SLM_output}")

# Specify output file paths
matched_entries_file = os.path.join(output_dir, "matched_entries.json")  # Same annotations
diff_LLM_file = os.path.join(output_dir, 'diff_LLM.json')  # Low confidence large model annotated sentences

# Output sentences with the same annotations
with open(matched_entries_file, "w") as file:
    json.dump(matched_entries, file, indent=4)
print(f"Data with the same annotations written to: {matched_entries_file}")

# Output sentences with low confidence large model annotations
with open(diff_LLM_file, "w") as file:
    json.dump(diff_LLM, file, indent=4)
print(f"Sentences with low confidence large model annotations written to: {diff_LLM_file}")

print("Data processing completed.")
