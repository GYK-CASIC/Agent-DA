import json
import re
import nltk
import random

# Build a dictionary of NER entity types
roles = {
    "Attacker": ["ORG", "PER", "GPE"],
    "Place": ["LOC", "GPE", "FAC"],
    "Target": ["LOC", "WEA", "PER", "FAC", "VEH", "ORG"],
    "Victim": ["PER"],
    "Agent": ["ORG", "PER", "GPE"],
    "Entity": ["ORG", "PER", "GPE"],
    "Instrument": ["WEA", "VEH"],
    "Artifact": ["WEA", "PER", "VEH", "FAC", "ORG"],
    "Origin": ["LOC", "GPE", "FAC"],
    "Vehicle": ["VEH"],
    "Destination": ["LOC", "GPE", "FAC"],
    "Buyer": ["ORG", "PER", "GPE"],
    "Person": ["PER"],
    "Org": ["ORG", "PER"],
    "Adjudicator": ["ORG", "PER", "GPE"],
    "Plaintiff": ["ORG", "PER", "GPE"],
    "Defendant": ["ORG", "PER", "GPE"],
    "Prosecutor": ["ORG", "PER", "GPE"],
    "Giver": ["ORG", "PER", "GPE"],
    "Seller": ["ORG", "PER", "GPE"],
    "Recipient": ["ORG", "PER", "GPE"],
    "Beneficiary": ["ORG", "PER", "GPE"],
    "Thing": ["WEA", "PER", "VEH", "FAC", "ORG"],
    "Audience": ["ORG", "PER", "GPE"],
}

input_data = ("processed_data/data_LLM")
output_file = ("processed_data/ace05e_dygieppformat/data.json")
with open(input_data, "r", encoding="utf-8") as file:
    data = json.load(file)

output_data = []
ner_info = []

for item in data:
    print("Current sentence", item)
    input_sentence = item["sentence"]
    events_info = item["events_info"]
    ner_info = []

    sentence_words = nltk.word_tokenize(input_sentence)

    events_per_data = []
    discard_entry = False
    for event_info in events_info:
        event_per_info = []
        input_sentence = item["sentence"]
        trigger_text = event_info["trigger_text"]
        event_type = event_info["event_type"]
        arguments = event_info["arguments"]

        trigger_start = None
        trigger_end = None
        for i, word in enumerate(sentence_words):
            word_regex = r'\b' + re.escape(word) + r'\b'  # Word boundary regex
            if re.search(word_regex, trigger_text):
                trigger_start = i
                trigger_end = i + len(word.split()) - 1
                break

        if trigger_start is None or trigger_end is None:
            print("Trigger word not found in current sentence", input_sentence)
            discard_entry = True
            break

        # Add event information
        event_per_info.append([trigger_start, event_type])

        # Calculate start and end positions of each argument
        for arg in arguments:
            arg_text = arg["text"]

            # Find the position of the argument in the word list
            arg_start = None
            arg_end = None
            arg_words = arg_text.split()
            for i, word in enumerate(sentence_words):
                word_regex = r'\b' + re.escape(word) + r'\b'  # Word boundary regex
                if re.search(word_regex, arg_text):
                    arg_start = i
                    arg_end = i + len(arg_words) - 1
                    break

            if arg_start is not None and arg_end is not None:
                arg_info = (
                arg_start, arg_end, arg["role"])  # Combine start position, end position, and role into a tuple
                event_per_info.append(arg_info)
                # Randomly select NER entity type and add to ner information
                entity_type = random.choice(roles[arg["role"]])
                ner_info.append([arg_start, arg_end, entity_type])
            else:
                discard_entry = True
                break

        if discard_entry:
            break

        # Add event to event list
        events_per_data.append(event_per_info)

    if discard_entry:
        continue

    # Construct output format for each data entry
    output_item = {
        "sentences": [sentence_words],
        "ner": [ner_info],
        "relations": [[]],
        "events": [events_per_data],
        "_sentence_start": [0],
        "doc_key": "CNN_CF_20030303.1900.02",
        "dataset": "ace-event",
        "clusters": [],
        "event_clusters": []
    }
    if output_item not in output_data:
        output_data.append(output_item)

# Output data in JSON format
with open(output_file, "w", encoding="utf-8") as output_file:
    for data_item in output_data:
        json.dump(data_item, output_file, ensure_ascii=False)
        output_file.write('\n')

print("Data has been successfully written to the file:", output_file)
