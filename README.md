# Agent-DA

## Environment Setup
```
1. conda activate DEGREE  # Environment for training DEGREE
2. conda activate DEGREE-master  # Environment for training the adjudicator
3. conda activate text2event  # Environment for training text2event
```

## Environment Requirements of DEGREE
- Python==3.8
- torch==1.8.0
- transformers==3.1.0 
- protobuf==3.17.3
- tensorboardx==2.4
- lxml==4.6.3
- beautifulsoup4==4.9.3
- bs4==0.0.1
- stanza==1.2
- sentencepiece==0.1.95
- ipdb==0.13.9

## Environment Requirements of DEGREE-master
- torch==1.9.0
- transformers==4.22.1
- datasets==2.4.0
- evaluate==0.2.2
- matplotlib==3.6.0
- rich==12.5.1
- scikit-learn==1.1.2
- requests==2.28.1

## Environment Requirements of text2event
- torch==1.7.1
- transformers==4.4.2
- anytree==2.8.0
- scikit-learn==1.1.2
- sacrebleu==1.4.14
- git-python==1.0.3
- elasticsearch
- nltk==3.5
- datasets>=1.1.3
- sentencepiece==0.1.92
- numpy==1.22
- tabulate==0.8.7
- filelock==3.0.12
- dataclasses==0.6
- rich==9.8.2

## Datasets
We support `ace05e` and `ace05ep`

### Preprocessing
Our preprocessing mainly adapts [OneIE's](https://blender.cs.illinois.edu/software/oneie/) released scripts with minor modifications. We deeply appreciate the contributions from the authors of the paper.

#### `ace05e`
1. Prepare data processed from [DyGIE++](https://github.com/dwadden/dygiepp#ace05-event)
2. Place the processed data into the folder `processed_data/ace05e_dygieppformat`
3. Run `scripts/process_ace05e.sh`

#### `ace05ep`
1. Download ACE data from [LDC](https://catalog.ldc.upenn.edu/LDC2006T06)
2. Run `scripts/process_ace05ep.sh`
The above scripts will generate processed data (including the full training set and the low-resource sets) in `./processed_data`

## Agent-DA for DEGREE
### Training DEGREE (End2end)
This step can be skipped because the training and inference can be performed simultaneously.

Use the following commands:

Generate data for DEGREE (End2end)
```bash
python degree/generate_data_degree_e2e.py -c config/config_degree_e2e_ace05ep.json
```
Train DEGREE (End2end)
```bash
python degree/train_degree_e2e1.py -c config/config_degree_e2e_ace05ep.json
```

The model will be stored at `./output/ACE05-ep/` by default.

### Three Steps of DA

#### Large Model Event Sample Generation
Due to limited budget, we perform sample generation using the ChatGPT web interface in a single-round dialogue format.

Run `generate_data.sh`

#### Collaborative Filtering
Use the samples generated by the LLM as a test set and input them into the SLM for prediction. In the file `DEGREE-masterbegin/RLHF/model.py`, we added confidence score calculation, so the model outputs confidence scores during prediction.

Then, filter the data based on whether the annotations are consistent and the size of the confidence score.
Run `Collaborative_Filtering/conf_scores.py`
- This part outputs initially filtered augmented samples and samples with confidence between 1-δ and δ, which need to be verified by the adjudicator.

#### Adjudicator-Assisted Filtering
Input samples with confidence between 0.05 and 0.95 (different annotations from large and small models) into the adjudicator for judgment. The file `RLHF/accuracy.txt` records the adjudicator's decision results, where 0 indicates the large model's annotation is better, and 1 indicates the small model's annotation is better.
Run `bash RLHF/judgement.sh`
- This part outputs the samples filtered by the adjudicator and then adds them to the previously filtered samples

### Constructing the Ranked Dataset and Training the Adjudicator
1. Run `RLHF/processcopy.py`
2. Run `RLHF/train_reward_model` to train the adjudicator

## Agent-DA for Text2event
### Training Text2event

```bash
bash run_seq2seq_with_pretrain.bash -d 0 -f tree -m model/t5-bash --label_smoothing 0 -l 1e-4 --lr_schedeler linear --warmup_steps 2000 -b 16
```

### Three Steps of Data Augmentation

#### Large Model Event Sample Generation
As shown in DEGREE

#### Large and Small Model Collaborative Filtering
1. Run `tree_to_event.py`
2. Run `confidence.py`

#### Adjudicator-Assisted Filtering
1. Run `RLHF/judgement.sh`
2. Run `RLHF/select_LLM_SLM.py`

## Acknowledgments and Citations
This project borrows or uses code from the following projects and resources, for which we are grateful:

- [DEGREE](https://github.com/PlusLabNLP/DEGREE) - Used as the event extraction model for our method. We validated our approach on DEGREE and followed DEGREE's data preprocessing steps.
- [Text2Event](https://github.com/luyaojie/Text2Event) - Used as the event extraction model for our method. We validated our approach on Text2Event.
- [RLHF](https://github.com/HarderThenHarder/transformers_tasks/tree/main/RLHF) - Referenced the RLHF project to construct a fine-grained event extraction ranking dataset and trained the adjudicator model.

We are very grateful for the contributions of the authors of the above projects.


If you find that the code is useful in your research, please consider citing our paper.

```@article{Tian and Guo2024Agent-DA,
  title={Agent-DA: Enhancing Low-Resource Event Extraction with Collaborative Multi-Agent Data Augmentation},
  author={Xuemeng Tian, Yikai Guo, Bin Ge, Xiaoguang Yuan, Hang Zhang, Yuting Yang, Wenjun Ke and Guozheng Li},
  year={2024}
}
```
