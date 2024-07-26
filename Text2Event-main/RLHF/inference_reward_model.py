#!/usr/bin/env python3
# Use the trained adjudicator to annotate large and small models
# Output annotations for choosing large or small models into a txt file
# python inference_reward_model.py

import torch
from rich import print
from transformers import AutoTokenizer, AutoModel, default_data_collator
from datasets import load_dataset
from functools import partial
from utils import convert_example
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to GPU.")
parser.add_argument("--max_seq_len", default=512, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")
args = parser.parse_args()  # Parse command line arguments

# device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained('RLHF/checkpoints/reward_model/sentiment_analysis/model_best/')
model = torch.load('RLHF/checkpoints/reward_model/sentiment_analysis/model_best/model.pt')
# model.to(device).eval()
model.to(args.device).eval()


def evaluate_model(model, data_loader):
    """
    Evaluate the training performance of the current model on the test set.

    Args:
        model: The current model
        data_loader: Dataloader for the test set
    """
    model.eval()
    with torch.no_grad():
        batch_rank_rewards = []
        for batch in data_loader:
            for batch_idx in range(len(batch['input_ids'])):
                rank_texts_count = len(batch['input_ids'][batch_idx])
                rank_rewards = []
                for text_idx in range(rank_texts_count):
                    reward = model(
                        batch['input_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['token_type_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['attention_mask'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['position_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                    )
                    # Compute the reward value for each sample
                    rank_rewards.append(reward[0])  # (rank_text_num) -> [tensor([0.1696]), tensor([0.3466])]
                # Add rank_rewards to the batch_rank_rewards list to form a 2D array
                batch_rank_rewards.append(
                    rank_rewards)  # (batch, rank_text_num) -> [[tensor([0.1696]), tensor([0.3466])], ...]
                # print(batch_rank_rewards)
                max_values_and_indices = [(max(row).item(), index) for index, row in enumerate(batch_rank_rewards)]
                max_values = [max(row).item() for row in batch_rank_rewards]

    # model.train()#
    total_ranklist, right_ranklist = 0, 0
    for rank_rewards in batch_rank_rewards:
        rank_rewards = [t.cpu().float() for t in rank_rewards]
        rank_rewards_sorted = sorted(rank_rewards, reverse=True)
        total_ranklist += 1
        if rank_rewards_sorted == rank_rewards:
            right_ranklist += 1
    return right_ranklist / total_ranklist, batch_rank_rewards


test_path = "RLHF/data/reward_datasets/sentiment_analysis/test.tsv"
dataset = load_dataset('text', data_files={'test': test_path})
# print(dataset)
convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
dataset = dataset.map(convert_func, batched=True)
test_dataset = dataset['test']

test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=16)
acc, reward = evaluate_model(model, test_dataloader)

max_values_and_indices = []

# Iterate through each sublist
for row in reward:
    max_tensor = max(row)  # Find the maximum tensor
    max_index = row.index(max_tensor)  # Determine the position of the maximum tensor
    max_values_and_indices.append((max_index))

print("Accuracy:", acc)
# Save results to a file
with open('RLHF/accuracy.txt', 'w') as file:
    # Write accuracy to the file
    for data in max_values_and_indices:
        file.write(f'{data}\n')

'''
texts = [
    'that is when american airlines flight 903 ran into trouble on a trip from boston to miami . trigger_text is trip, event_type is Movement:Transport, text: "american airlines", role: "Agent",text:"flight 903",role:"Vehicle",text:"boston",role:"Origin",text:"miami",role:"Destination"',
    'that is when american airlines flight 903 ran into trouble on a trip from boston to miami . trigger_text is trip, event_type is Life：Die, text: "american airlines", role: "Agent",text:"flight 903",role:"Vehicle",text:"boston",role:"Origin"',
    'that is when american airlines flight 903 ran into trouble on a trip from boston to miami . trigger_text is is, event_type is Movement:Transport, text: "american airlines", role: "Agent",text:"flight 903",role:"Vehicle",text:"boston",role:"Origin",text:"miami",role:"Destination"'
]
inputs = tokenizer(
    texts, 
    max_length=128,
    padding='max_length', 
    return_tensors='pt'
)
r = model(**inputs)
print(r)
'''
