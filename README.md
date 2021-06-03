# TPT-Summ-dev
This repo contains the source code of the TP-Transformer model (TPT-d) described in the following paper 
* *"Enriching Transformers with Structured Tensor-Product Representations for Abstractive Summarization"* in Proceedings of NAACL-HLT, 2021. ([paper](https://www.aclweb.org/anthology/2021.naacl-main.381.pdf)).

The basic code structure was adapted from the HuggingFace [Transformers](https://github.com/huggingface/transformers).

## 0. Preparation
### 0.1 Dependencies
* PyTorch 1.4.0/1.6.0/1.8.0
* See `requirements.txt`.

### 0.2 Data
* Download and preprocess the [XSum](https://github.com/EdinburghNLP/XSum) dataset and put it under the ```data``` folder, which should a structure like this:
```
--- TPT-Summ
------ data
--------- xsum
------------ train.source
------------ train.target
------------ val.source
------------ val.target
------------ test.source
------------ test.target
```
* Other dataset should follow the same structure.


## 1. Training a TP-Transformer on XSum Dataset
* Every experiment has a unique id `run_id`, which should be consistent throughout training and evaluation. The default `run_id` 
in `./train_scripts/train_tpt_xsum.sh` is `00`.
* Train a TP-Transformer on `XSum` dataset by running the training script. 
```
./train_scripts/train_tpt_xsum.sh
```
* The Tensorboard log files are saved in `out/xsum/[RUN_ID]/log`.
* During the training, we calculate the dev-set ROUGE scores after every epoch and log them in the Tensorboard logs. The ROUGE scores 
are calculated using `https://github.com/pltrdy/rouge`, which is a simplified version of the official ROUGE package. 
Therefore, the scores here are lower than those reported in the paper. 
We will explain how to get the official ROUGE scores in the next section.

## 2. Generating Summaries
* Update the identifiers and parameters in `eval_scripts/eval_tpt_xsum.sh`:
    * Update the `run_id` to match the model you want to evaluate (defalut is 00）.
    * Update `evaluate_epoch`. The default value is 30, which is where we got our best model.
    * Update `num_beams` to choose between beam search and greedy search. Default is greedy search (`seed=1`).
* Decode the TP-Transformer on `XSum` validation set:
```
./eval_scripts/eval_tpt_xsum.sh
```

## 3. Compute the ROUGE scores
* Install the [Files2ROUGE](https://github.com/pltrdy/files2rouge) package in order to calculate the ROUGE scores of generated summaries.
* Obtain the ROUGE scores of the generated summaries for the model `run_id=00`:
```
files2rouge data/xsum/val.target out/xsum/00/xsum/epoch=30_beam=1_generated_summaries.txt
```

## Citation
```
@inproceedings{jiang-etal-2021-enriching,
    title = "Enriching Transformers with Structured Tensor-Product Representations for Abstractive Summarization",
    author = "Jiang, Yichen  and
      Celikyilmaz, Asli  and
      Smolensky, Paul  and
      Soulos, Paul  and
      Rao, Sudha  and
      Palangi, Hamid  and
      Fernandez, Roland  and
      Smith, Caitlin  and
      Bansal, Mohit  and
      Gao, Jianfeng",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.381",
    pages = "4780--4793",
}