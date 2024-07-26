import logging
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForPreTraining
import ipdb
import math 
logger = logging.getLogger(__name__)
import torch, gc

gc.collect()
torch.cuda.empty_cache()

class GenerativeModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        logger.info(f'Loading pre-trained model {config.model_name}')
        self.model_config =  AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        self.model = AutoModelForPreTraining.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, batch):
        outputs = self.model(input_ids=batch.enc_idxs, 
                             attention_mask=batch.enc_attn, 
                             decoder_input_ids=batch.dec_idxs, 
                             decoder_attention_mask=batch.dec_attn, 
                             labels=batch.lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        
        return loss
        
    def predict(self, batch, num_beams=4, max_length=50):
        self.eval()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=batch.enc_idxs, 
                                          attention_mask=batch.enc_attn, 
                                          num_beams=num_beams, 
                                          max_length=max_length)
            # The returned output is of the type beamBeamSearchEncoderDecoderOutput, which includes sequence scores and other information.
            output = self.model.generate(input_ids=batch.enc_idxs,
                                          attention_mask=batch.enc_attn,
                                          num_beams=num_beams,
                                          return_dict_in_generate=True,
                                          output_scores=True,
                                          max_length=max_length)
            
        final_output = []
        confidence_scores = []
        for bid in range(len(batch.enc_idxs)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
            conf_scores = output.sequences_scores[bid]
            
            conf_scores = math.exp(conf_scores)*100
            conf_scores = round(conf_scores, 2)
            
            #conf_scores_percentage = round(math.exp(conf_scores)*100, 2)
            # conf_scores = conf_scores_percentage
            confidence_scores.append(conf_scores)
            # print("out_sentence", output_sentence,"confidence_scores",f"{conf_scores}%")
        self.train()
        return final_output,confidence_scores

