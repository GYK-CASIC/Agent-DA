# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
Utility class.
"""
import traceback

import numpy as np
from rich import print


def convert_example(examples: dict, tokenizer, max_seq_len: int):
    """
    Convert sample data into input data accepted by the model.

    Args:
        examples (dict): Training data samples, e.g. -> {
                                                "text": [
                                                            'sentence1	sentence2	sentence3',
                                                            'sentence1	sentence2	sentence3',
                                                            ...
                                                ]
                                            }

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [
                                            [[101, 3928, ...], [101, 4395, ...], [101, 2135, ...]],
                                            [[101, 3928, ...], [101, 4395, ...], [101, 2135, ...]],
                                            ...
                                        ], 
                            'token_type_ids': [
                                                [[0, 0, ...], [0, 0, ...], [0, 0, ...]],
                                                [[0, 0, ...], [0, 0, ...], [0, 0, ...]],
                                                ...
                                            ]
                            'position_ids': [
                                                [[0, 1, 2, ...], [0, 1, 2, ...], [0, 1, 2, ...]],
                                                [[0, 1, 2, ...], [0, 1, 2, ...], [0, 1, 2, ...]],
                                                ...
                                            ]
                            'attention_mask': [
                                                [[1, 1, ...], [1, 1, ...], [1, 1, ...]],
                                                [[1, 1, ...], [1, 1, ...], [1, 1, ...]],
                                                ...
                                            ]
                        }
    """
    tokenized_output = {
        'input_ids': [], 
        'token_type_ids': [],
        'position_ids': [],
        'attention_mask': []
    }

    for example in examples['text']:
        try:
            rank_texts = example.strip().split('\t')
        except:
            print(f'Error processing example: "{example}"')
            continue
        
        # DEBUG: Output the length of each sample
        print([len(tokenizer.encode(text)) for text in rank_texts])
        
        rank_texts_prop = {
            'input_ids': [], 
            'token_type_ids': [],
            'position_ids': [],
            'attention_mask': []
        }
        for rank_text in rank_texts:
            encoded_inputs = tokenizer(
                    text=rank_text,
                    truncation=True,
                    max_length=max_seq_len,
                    padding='max_length')
            rank_texts_prop['input_ids'].append(encoded_inputs["input_ids"])
            rank_texts_prop['token_type_ids'].append(encoded_inputs["token_type_ids"])
            rank_texts_prop['position_ids'].append([i for i in range(len(encoded_inputs["input_ids"]))])
            rank_texts_prop['attention_mask'].append(encoded_inputs["attention_mask"])

        for k, v in rank_texts_prop.items():
            tokenized_output[k].append(v)
    
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)
    
    return tokenized_output