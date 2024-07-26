import os
import sys
import json
import logging
import time
import pprint
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import GenerativeModel
from data import GenDataset
from utils import Summarizer, compute_f1
from argparse import ArgumentParser, Namespace

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config.update(args.__dict__)
config = Namespace(**config)

if config.dataset == "ace05e" or config.dataset == "ace05ep":
    from template_generate_ace import eve_template_generator
    template_file = "template_generate_ace"
elif config.dataset == "ere":
    from template_generate_ere import eve_template_generator
    template_file = "template_generate_ere"

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# logger and summarizer
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "test.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
summarizer = Summarizer(output_dir)

# set GPU device
torch.cuda.set_device(config.gpu_device)

# check valid styles
assert np.all([style in ['event_type_sent', 'keywords', 'template'] for style in config.input_style])
assert np.all([style in ['trigger:sentence', 'argument:sentence'] for style in config.output_style])

# output
with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
    json.dump(vars(config), fp, indent=4)
best_model_path = os.path.join(output_dir, 'best_model.mdl')
test_prediction_path = os.path.join(output_dir, 'pred.test.json')

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
special_tokens = ['<Trigger>', '<sep>']
tokenizer.add_tokens(special_tokens)

# load data
test_set = GenDataset(tokenizer, config.max_length, config.test_finetune_file, config.max_output_length)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

# initialize and load the model
model = GenerativeModel(config, tokenizer)
model.load_state_dict(torch.load("output/ACE05-ep/20240715_085420/best_model.mdl", map_location=f'cuda:{config.gpu_device}'))
model.cuda(device=config.gpu_device)
model.eval()

# start testing
logger.info("Start testing ...")
progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test')
best_test_flag = False
write_output = []
test_gold_tri_num, test_pred_tri_num, test_match_tri_num = 0, 0, 0
test_gold_arg_num, test_pred_arg_num, test_match_arg_id, test_match_arg_cls = 0, 0, 0, 0

for batch_idx, batch in enumerate(DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn)):
    progress.update(1)
    pred_text, confidence_scores = model.predict(batch, max_length=config.max_output_length)
    gold_text = batch.target_text
    input_text = batch.input_text
    for i_text, g_text, p_text, info, conf_scores in zip(input_text, gold_text, pred_text, batch.infos, confidence_scores):
        theclass = getattr(sys.modules[template_file], info[1].replace(':', '_').replace('-', '_'), False)
        assert theclass
        template = theclass(config.input_style, config.output_style, info[2], info[1], info[0])
        pred_object = template.decode(p_text)
        gold_object = template.trigger_span + [_ for _ in template.get_converted_gold()]
        
        # calculate scores
        sub_scores = template.evaluate(pred_object)
        test_gold_tri_num += sub_scores['gold_tri_num']
        test_pred_tri_num += sub_scores['pred_tri_num']
        test_match_tri_num += sub_scores['match_tri_num']
        test_gold_arg_num += sub_scores['gold_arg_num']
        test_pred_arg_num += sub_scores['pred_arg_num']
        test_match_arg_id += sub_scores['match_arg_id']
        test_match_arg_cls += sub_scores['match_arg_cls']
        write_output.append({
            'input text': i_text,
            'gold text': g_text,
            'pred text': p_text,
            'gold triggers': gold_object,
            'pred triggers': pred_object,
            'score': sub_scores,
            'gold info': info,
            'confidence scores': conf_scores
        })
progress.close()

test_scores = {
    'tri_id': compute_f1(test_pred_tri_num, test_gold_tri_num, test_match_tri_num),
    'arg_id': compute_f1(test_pred_arg_num, test_gold_arg_num, test_match_arg_id),
    'arg_cls': compute_f1(test_pred_arg_num, test_gold_arg_num, test_match_arg_cls)
}

# print scores
print("---------------------------------------------------------------------")
print('Trigger I  - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    test_scores['tri_id'][0] * 100.0, test_match_tri_num, test_pred_tri_num, 
    test_scores['tri_id'][1] * 100.0, test_match_tri_num, test_gold_tri_num, test_scores['tri_id'][2] * 100.0))
print("---------------------------------------------------------------------")
print('Role I     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    test_scores['arg_id'][0] * 100.0, test_match_arg_id, test_pred_arg_num, 
    test_scores['arg_id'][1] * 100.0, test_match_arg_id, test_gold_arg_num, test_scores['arg_id'][2] * 100.0))
print('Role C     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    test_scores['arg_cls'][0] * 100.0, test_match_arg_cls, test_pred_arg_num, 
    test_scores['arg_cls'][1] * 100.0, test_match_arg_cls, test_gold_arg_num, test_scores['arg_cls'][2] * 100.0))
print("---------------------------------------------------------------------")

with open(test_prediction_path, 'w') as fp:
    json.dump(write_output, fp, indent=4)

logger.info({"test_scores": test_scores})
logger.info("Testing done!")
