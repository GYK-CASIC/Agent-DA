import os, sys, json, logging, time, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from model import GenerativeModel
from data import GenDataset
from utils import Summarizer, compute_f1
from argparse import ArgumentParser, Namespace
import ipdb

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
fixed_subdir = "my_output"
output_dir = os.path.join(config.output_dir, fixed_subdir)
# output_dir = config.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
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
config_path = os.path.join(output_dir, 'config.json')
best_model_path = os.path.join(output_dir, 'best_model.mdl')
test_prediction_path = os.path.join(output_dir, 'pred.test.json')

with open(config_path, 'w') as fp:
    json.dump(vars(config), fp, indent=4)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
special_tokens = ['<Trigger>', '<sep>']
tokenizer.add_tokens(special_tokens)

# load data
train_set = GenDataset(tokenizer, config.max_length, config.train_finetune_file, config.max_output_length)
# dev_set = GenDataset(tokenizer, config.max_length, config.dev_finetune_file, config.max_output_length)
test_set = GenDataset(tokenizer, config.max_length, config.test_finetune_file, config.max_output_length)
train_batch_num = len(train_set) // config.train_batch_size + (len(train_set) % config.train_batch_size != 0)
# dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = GenerativeModel(config, tokenizer)
model.cuda(device=config.gpu_device)

# optimizer
param_groups = [{'params': model.parameters(), 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=train_batch_num*config.warmup_epoch,
                                           num_training_steps=train_batch_num*config.max_epoch)

logger.info("Start training ...")
summarizer_step = 0
best_test_epoch = -1
best_test_scores = {
    'tri_id': (0.0, 0.0, 0.0),
    'arg_id': (0.0, 0.0, 0.0),
    'arg_cls': (0.0, 0.0, 0.0)
}
for epoch in range(1, config.max_epoch+1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")
    
    # training
    progress = tqdm.tqdm(total=train_batch_num, ncols=75, desc='Train {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(DataLoader(train_set, batch_size=config.train_batch_size // config.accumulate_step, 
                                                 shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
        
        # forard model
        loss = model(batch)
        
        # record loss
        summarizer.scalar_summary('train/loss', loss, summarizer_step)
        summarizer_step += 1
        loss = loss * (1 / config.accumulate_step)
        loss.backward()
        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    progress.close()
    if epoch >= 45:
        # eval test 
        progress = tqdm.tqdm(total=test_batch_num,  ncols=75, desc='Test {}'.format(epoch))
        model.eval()
        best_test_flag = False
        write_output = []#  Used to store the model's predictions and related information on the validation set
        test_gold_tri_num, test_pred_tri_num, test_match_tri_num = 0, 0, 0

        test_gold_arg_num, test_pred_arg_num, test_match_arg_id, test_match_arg_cls = 0, 0, 0, 0
        for batch_idx, batch in enumerate(DataLoader(test_set, batch_size=config.eval_batch_size,
                                                 shuffle=False, collate_fn=test_set.collate_fn)):
            progress.update(1)
            # Use the model to infer on the validation set, obtain predicted text and confidence scores
            pred_text, confidence_scores = model.predict(batch, max_length=config.max_output_length)
            gold_text = batch.target_text
            input_text = batch.input_text
            # Iterate through each sample, processing input text i_text, g_text, predicted text, related information, and confidence scores
            for i_text, g_text, p_text, info, conf_scores in zip(input_text, gold_text, pred_text, batch.infos, confidence_scores):
            # 获取模板类
                theclass = getattr(sys.modules[template_file], info[1].replace(':', '_').replace('-', '_'), False)
                assert theclass
                template = theclass(config.input_style, config.output_style, info[2], info[1], info[0])
                # decode predictions
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
        # check best dev model
        if test_scores['arg_cls'][2] > best_test_scores['arg_cls'][2]:
            best_test_flag = True
        
        # if best dev, save model and evaluate test set
        if best_test_flag:    
            best_test_scores = test_scores
            best_test_epoch = epoch
        
            # save best model
            logger.info('Saving best model')
            torch.save(model.state_dict(), best_model_path)
        
            # save dev result
            with open(test_prediction_path, 'w') as fp:
                json.dump(write_output, fp, indent=4)

        logger.info({"epoch": epoch, "test_scores": test_scores})
        if best_test_flag:
            logger.info({"epoch": epoch, "test_scores": test_scores})
        logger.info("Current best")
        logger.info({"best_epoch": best_test_epoch, "best_scores": best_test_scores})
        
logger.info(log_path)
logger.info("Done!")

