# Process the data
# python tree_to_event.py
import json

def parse_tree_structure(tree_data):
    all_events = []

    for line in tree_data.splitlines():
        events = {}
        events["sentence"] = ""
        event_list = []

        segments = line.strip().split("<extra_id_0>")[2:]  # Remove the first empty element

        for segment in segments:
            segment = segment.strip().replace("<extra_id_1>", "").strip()
            parts = segment.split(maxsplit=1)
            remainder = parts[1] if len(parts) > 1 else None
            first_word = parts[0]

            event_type_mapping = {
                "Sentence": "Justice:Sentence",
                "Convict": "Justice:Convict",
                "Trial-Hearing": "Justice:Trial-Hearing",
                "End-Position": "Personnel:End-Position",
                "Execute": "Justice:Execute",
                "Extradite": "Justice:Extradite",
                "Phone-Write": "Contact:Phone-Write",
                "Start-Position": "Personnel:Start-Position",
                "Acquit": "Justice:Acquit",
                "Demonstrate": "Conflict:Demonstrate",
                "Sue": "Justice:Sue",
                "Release-Parole": "Justice:Release-Parole",
                "Arrest-Jail": "Justice:Arrest-Jail",
                "Be-Born": "Life:Be-Born",
                "Declare-Bankruptcy": "Business:Declare-Bankruptcy",
                "Pardon": "Justice:Pardon",
                "Meet": "Contact:Meet",
                "Appeal": "Justice:Appeal",
                "Transfer-Money": "Transaction:Transfer-Money",
                "End-Org": "Business:End-Org",
                "Transfer-Ownership": "Transaction:Transfer-Ownership",
                "Start-Org": "Business:Start-Org",
                "Nominate": "Personnel:Nominate",
                "Marry": "Life:Marry",
                "Fine": "Justice:Fine",
                "Transport": "Movement:Transport",
                "Charge-Indict": "Justice:Charge-Indict",
                "Die": "Life:Die",
                "Merge-Org": "Business:Merge-Org",
                "Elect": "Personnel:Elect",
                "Injure": "Life:Injure",
                "Attack": "Conflict:Attack",
                "Divorce": "Life:Divorce"
            }

            if first_word in event_type_mapping:
                trigger_text = remainder
                event = {
                    'trigger_text': trigger_text,
                    'event_type': event_type_mapping[first_word],
                    'argument': []
                }
                event_list.append(event)
            else:
                role = first_word
                text = remainder if remainder is not None else 'null'
                if event_list:
                    event_list[-1]['argument'].append({'role': role, 'text': text})

        events['events_info'] = event_list
        all_events.append(events)
    
    return all_events

# Paths to the files
tree_file_path = 'models/001/event_finetune/test_preds_seq2seq.txt'

with open(tree_file_path, 'r', encoding='utf-8') as tree_file:
    tree_data = tree_file.read()

# Parse tree structure and update JSON data
events_info = parse_tree_structure(tree_data)

# Write events_info to json file
with open('processDATA/events_info.json', 'w', encoding='utf-8') as output_file:
    json.dump(events_info, output_file, indent=4)

# Load data1
with open('processDATA/data.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

# Replace events_info in data1
def replace_events_info(data1, events_info):
    for i in range(len(data1)):
        data1[i]['events_info'] = events_info[i]['events_info']
    return data1

updated_data = replace_events_info(data1, events_info)

# Write updated data to json file
with open('processDATA/updated_data.json', 'w', encoding='utf-8') as f:
    json.dump(updated_data, f, indent=4)
