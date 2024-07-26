import json
with open("RLHF/diff_SLM.json") as file:
    data = json.load(file)
    
sentences = []
single_data = []
for entry in data:
    if entry["sentence"] not in sentences:
            sentences.append(entry["sentence"])
            single_data.append(entry)
output = "RLHF/diff_SLM.json"
with open(output, "w") as file:
    json.dump(single_data, file,indent=4)
print("done",output)
