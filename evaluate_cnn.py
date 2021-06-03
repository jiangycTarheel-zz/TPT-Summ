"""
Decode summarization models (generate summaries) trained with finetune.py
Multi-GPU decoding not working yet.
"""

import argparse
import os
import logging
import glob
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler
from rouge import Rouge
from tqdm import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from models.modeling_tpt import TPTForConditionalGeneration
from models.configuration_tpt import TPTConfig
from utils import SummarizationDataset

logger = logging.getLogger(__name__)

MODELS = {
    "t5": T5ForConditionalGeneration,
    "tp": TPTForConditionalGeneration,
}

CONFIGS = {
    "t5": T5Config,
    "tp": TPTConfig,
}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries(lns, args):
    args.batch_size = args.batch_size * max(1, args.n_gpu)

    logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
    checkpoints = list(sorted(glob.glob(os.path.join(args.model_path, "checkpointepoch="+str(args.evaluate_epoch)+".ckpt"), recursive=True)))
    print(os.path.join(args.model_path, "checkpointepoch="+str(args.evaluate_epoch)+".ckpt"))
    checkpoint = checkpoints[0]

    logger.info("Evaluate the following checkpoint: %s", checkpoint)
    num_epoch = checkpoint.split("epoch=")[1].split(".ckpt")[0]

    if args.role_attn_vis:
        dec_role_attn_output_path = os.path.join(args.output_path, 'epoch=' + num_epoch + '_beam=' + str(args.num_beams) \
                                                  + '_dec_role_attn.txt')
        enc_role_attn_output_path = os.path.join(args.output_path,
                                                      'epoch=' + num_epoch + '_beam=' + str(args.num_beams) \
                                                      + '_enc_role_attn.txt')
        dec_role_attn_output_file = Path(dec_role_attn_output_path).open("w", encoding='utf-8')
        enc_role_attn_output_file = Path(enc_role_attn_output_path).open("w", encoding='utf-8')

    args.score_path = os.path.join(args.output_path, 'epoch=' + num_epoch + '_beam=' + str(args.num_beams) + '_' + args.score_path)
    args.output_path = os.path.join(args.output_path, 'epoch=' + num_epoch + '_beam=' + str(args.num_beams) + '_' + args.output_filename)
    #if os.path.exists(args.output_path):
    #    raise ValueError(
    #        "Output file ({}) already exists and is not empty.".format(
    #            args.output_path
    #        )
    #    )
    output_file = Path(args.output_path).open("w", encoding='utf-8')

    # Reload the model
    config = CONFIGS[args.model_name_or_path[:2]].from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    config.num_roles = 50
    tokenizer = T5Tokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    model = MODELS[args.model_name_or_path[:2]].from_pretrained(
        args.model_name_or_path,
        return_unpretrained=True,
        config=config,
        cache_dir=args.cache_dir,
    )

    # Restore the model parameters
    state_dict_pl = torch.load(checkpoint)["state_dict"]
    state_dict = {}
    for weight in state_dict_pl:
        if 'label_smoothing' in weight or 'rq' in weight or 'rk' in weight:
            continue
        if weight.startswith("model."):
            state_dict[weight[6:]] = state_dict_pl[weight]
        else:
            state_dict[weight] = state_dict_pl[weight]
    model.load_state_dict(state_dict)
    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    dataset = SummarizationDataset(
        tokenizer, 
        data_dir=args.data_dir,
        type_path=args.data_split,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    eval_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    # update config with summarization specific params
    if args.n_gpu > 1:
        task_specific_params = model.module.config.task_specific_params
        if task_specific_params is not None:
            model.module.config.update(task_specific_params.get("summarization", {}))
    else:
        task_specific_params = model.config.task_specific_params
        if task_specific_params is not None:
            model.config.update(task_specific_params.get("summarization", {}))

    epoch_iterator = tqdm(dataloader, desc="Iteration", disable=False)
    for step, batch in enumerate(epoch_iterator):
        model.eval()
        input_ids = batch["source_ids"].to(args.device)
        attention_mask = batch["source_mask"].to(args.device)

        #batch = tuple(batch[t].to(args.device) for t in batch)
        #input_ids, attention_mask = batch[0], batch[1]
        if args.n_gpu > 1:
            summaries = model.module.generate(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              num_beams=args.num_beams,
                                              role_attn_vis=args.role_attn_vis,
                                              max_length=args.max_length,
                                              min_length=args.min_length)
        else:
            summaries = model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       num_beams=args.num_beams,
                                       role_attn_vis=args.role_attn_vis,
                                       max_length=args.max_length,
                                       min_length=args.min_length)

            if args.role_attn_vis:
                #enc_self_role_attention = model.encoder.block[-1].layer[0].SelfAttention.role_weights.data.cpu().numpy()
                dec_self_role_attention = model.dec_self_role_attn.data.cpu().numpy()
                dec_cross_role_attention = model.dec_cross_role_attn.data.cpu().numpy()

                #enc_self_max_roles = np.argsort(-enc_self_role_attention, axis=-1)[:, :, :, 0]
                dec_self_max_roles = np.argsort(-dec_self_role_attention, axis=-1)[:, :, :, 0]
                dec_cross_max_roles = np.argsort(-dec_cross_role_attention, axis=-1)[:, :, :, 0]

                dec_self_valid_roles = [np.where(r > 0.1) for r in list(dec_self_role_attention[0, 0])]
                dec_cross_valid_roles = [np.where(r > 0.1) for r in list(dec_cross_role_attention[0, 0])]

        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
        token_ids = [tokenizer.convert_ids_to_tokens(g, skip_special_tokens=True) for g in summaries]
        source = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in input_ids]
        input_tokens = [tokenizer.convert_ids_to_tokens(d, skip_special_tokens=True) for d in input_ids]

        for ih, hypothesis in enumerate(dec):
            output_file.write(hypothesis + "\n")
            output_file.flush()
            if args.role_attn_vis:
                dec_role_attn_str, enc_role_attn_str = [], []
                for w, r in zip(token_ids[ih], np.transpose(dec_cross_max_roles[ih])):
                    #role_attn_str.append('(' + w + ',' + str(r[0]) + ',' + str(r[1]) + ',' + str(r[2]) + ',' + str(r[3]) \
                    #                     + ',' + str(r[4]) + ',' + str(r[5]) + ',' + str(r[6]) + ',' + str(r[7]) + ')')
                    dec_role_attn_str.append('(' + w + ',' + str(r[0]) + ')')
                dec_role_attn_output_file.write(' '.join(dec_role_attn_str) + '\n')
                dec_role_attn_output_file.write(hypothesis + "\n")
                dec_role_attn_output_file.flush()


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="T5 model size, either 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'. Defaults to 't5-base'.",
        default="t5-base",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--n_gpu", type=int, default=1, help="",
    )
    parser.add_argument(
        "--max_source_length", type=int, default=512, help="",
    )
    parser.add_argument(
        "--max_target_length", type=int, default=56, help="",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./cache", help="",
    )
    parser.add_argument(
        "--model_path", type=str, default="out/t5_sum", help="the location of the model to be eval.",
    )
    parser.add_argument(
        "--run_id", type=str, default='00', help="",
    )
    parser.add_argument(
        "--evaluate_epoch", type=int, default=-1, help="",
    )
    parser.add_argument(
        "--dataset_name", default="cnn_dm", type=str, help="The data to evaluate on.",
    )
    parser.add_argument(
        "--data_split", default="val", type=str, help="The data to evaluate on.",
    )
    parser.add_argument(
        "--eval_dataset_name", type=str, help="The data to evaluate on.",
    )
    parser.add_argument(
        "--data_dir", default="./data", type=str, help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.",
    )
    parser.add_argument(
        "--input_path", type=str, default="val.source", help="like cnn_dm/test_articles_input.txt",
    )
    parser.add_argument(
        "--reference_path", type=str, default="val.target", help="like cnn_dm/test_reference_summaries.txt"
    )
    parser.add_argument(
        "--output_path", type=str, help="where to save summaries",
    )
    parser.add_argument(
        "--output_filename", type=str, default="generated_summaries.txt", help="where to save summaries",
    )
    parser.add_argument(
        "--score_path", type=str, default="rouge_scores.txt", help="where to save the rouge score",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, required=False, help="batch size: how many to summarize at a time",
    )
    parser.add_argument(
        "--no_cuda", default=False, type=bool, help="Whether to force the execution on CPU.",
    )
    parser.add_argument(
        "--num_beams", default=1, type=int, help="Beam size."
    )
    parser.add_argument(
        "--max_length", default=200, type=int, help="The max length of generated summaries."
    )
    parser.add_argument(
        "--min_length", default=30, type=int, help="The min length of generated summaries."
    )
    parser.add_argument(
        "--role_attn_vis", action="store_true",
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if args.eval_dataset_name is None:
        args.eval_dataset_name = args.dataset_name
    args.model_path = os.path.join(args.model_path, args.dataset_name, args.run_id)
    if args.output_path is None:
        args.output_path = os.path.join(args.model_path, args.eval_dataset_name)
    else:
        args.output_path = os.path.join(args.output_path, args.eval_dataset_name)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.data_dir = os.path.join(args.data_dir, args.eval_dataset_name)
    args.input_path = os.path.join(args.data_dir, args.input_path)
    args.reference_path = os.path.join(args.data_dir, args.reference_path)
    # if args.output_path is None:
    #     args.output_path = args.model_path
    # else:
    #     args.output_path = os.path.join(args.output_path, args.eval_dataset_name, args.run_id)
    source_lns = [x.rstrip() for x in open(args.input_path, "r", encoding='utf-8').readlines()]

    generate_summaries(source_lns, args)


if __name__ == "__main__":
    run_generate()