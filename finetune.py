# Adapted from the HuggingFace Transformers directory: https://github.com/huggingface/transformers/.


import argparse
import glob
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

try:
    from .utils import SummarizationDataset, calculate_rouge, LabelSmoothingLoss
    from .optim_utils import get_inverse_sqrt_schedule_with_warmup
except ImportError:
    from utils import SummarizationDataset, calculate_rouge, LabelSmoothingLoss
    from optim_utils import get_inverse_sqrt_schedule_with_warmup

logger = logging.getLogger(__name__)


class SummarizationTrainer(BaseTransformer):

    mode = "language-modeling"

    def __init__(self, hparams):
        config_kwargs = {}
        if "tpt" in hparams.model_name_or_path:
            self.mode = "tpt-language_modeling"
            config_kwargs: dict = dict(
                num_roles=hparams.num_roles_per_layer,
            )
        super().__init__(hparams, num_labels=None, mode=self.mode, **config_kwargs)
        # The tokenizer is initialized in BaseTransformer from AutoTokenizer, but shortcut it here to
        # avoid making our tokenizer AutoTokenizer capable.
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
        )
        if self.hparams.label_smooth > 0:
            self.label_smoothing = LabelSmoothingLoss(self.hparams.label_smooth,
                                                      self.config.vocab_size,
                                                      ignore_index=self.tokenizer.pad_token_id)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
        return self.model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels,
        )

    def _step(self, batch, data_type):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        if self.hparams.dont_prepend_bos:
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        else:
            lm_labels = y.clone()
            lm_labels[y[:, :] == pad_token_id] = -100
            outputs = self(source_ids, attention_mask=source_mask, lm_labels=lm_labels)

        if self.hparams.label_smooth > 0:
            logits = outputs[1]
            loss = self.label_smoothing(logits, y)
        else:
            loss = outputs[0]

        losses = {}
        overall_loss = loss.clone()

        if self.mode == 'tpt-language_modeling' and self.config.use_discrete_roles and self.hparams.role_regularization > 0:
            reg_loss = self.hparams.role_regularization * outputs[-7]
            overall_loss += reg_loss
            losses[data_type + '_regularization_loss'] = reg_loss.clone()

        losses[data_type + '_cross_entropy_loss'] = loss.clone()
        losses[data_type + '_loss'] = overall_loss.clone()

        if self.mode == 'tpt-language_modeling' and not self.training and self.config.use_discrete_roles:
            batch_num_elements_above_98, batch_num_elements_above_90, \
                batch_num_elements, batch_encoder_layerwise_roles_used, batch_decoder_self_layerwise_roles_used, \
                batch_decoder_enc_layerwise_roles_used = outputs[-6:]
            losses['elements_above_98'] = batch_num_elements_above_98
            losses['elements_above_90'] = batch_num_elements_above_90
            losses['num_elements'] = batch_num_elements

        return overall_loss, losses

    def training_step(self, batch, batch_idx):
        loss, tensorboard_logs = self._step(batch, data_type='train')
        #tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    #def validation_step(self, batch, batch_idx):
        #loss = self._step(batch)
        #return {"val_loss": loss}

    def validation_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = SummarizationDataset.trim_seq2seq_batch(
            batch,
            pad_token_id,
            trim_y=True,
        )
        # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_cnn.py
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=1,
            max_length=80,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )

        # preds = [
        #     self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        #     for g in generated_ids
        # ]
        # target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in y]
        loss, tensorboard_logs = self._step(batch, data_type='val')

        tensorboard_logs["preds"] = generated_ids
        tensorboard_logs["target"] = y
        return tensorboard_logs

    def _validation_end(self, outputs):
        tensorboard_logs = {}
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_cross_entropy_loss = torch.stack([x["val_cross_entropy_loss"] for x in outputs]).mean()
        tensorboard_logs["val_loss"] = avg_loss
        tensorboard_logs["val_cross_entropy_loss"] = avg_cross_entropy_loss

        if self.mode == 'tpt-language_modeling' and "num_elements" in outputs[0]:
            total_num_elements = torch.stack([x["num_elements"] for x in outputs]).sum()
            total_elements_above_98 = torch.stack([x["elements_above_98"] for x in outputs]).sum()
            total_elements_above_90 = torch.stack([x["elements_above_90"] for x in outputs]).sum()
            tensorboard_logs["elements_above_90"] = total_elements_above_90.item() / total_num_elements.item()
            tensorboard_logs["elements_above_98"] = total_elements_above_98.item() / total_num_elements.item()
            if self.mode == 'tpt-language_modeling' and self.config.use_discrete_roles and self.hparams.role_regularization > 0:
                avg_reg_loss = torch.stack([x["val_regularization_loss"] for x in outputs]).mean()
                tensorboard_logs["val_regularization_loss"] = avg_reg_loss

        rouge1 = torch.stack([x["rouge-1"] for x in outputs]).mean()
        rouge2 = torch.stack([x["rouge-2"] for x in outputs]).mean()
        rougel = torch.stack([x["rouge-l"] for x in outputs]).mean()
        tensorboard_logs["rouge-1"], tensorboard_logs["rouge-2"], tensorboard_logs["rouge-l"] = rouge1, rouge2, rougel
        return {"avg_val_loss": avg_loss, "rouge-1": rouge1, "rouge-2": rouge2, "rouge-l": rougel, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions.txt")
        output_test_targets_file = os.path.join(self.hparams.output_dir, "test_targets.txt")
        # write predictions and targets for later rouge evaluation.
        with open(output_test_predictions_file, "w+", encoding='utf-8') as p_writer, \
                open(output_test_targets_file, "w+", encoding='utf-8') as t_writer:
            for output_batch in outputs:
                predictions = [
                    self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for g in output_batch["preds"]
                ]
                gts = [
                    self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for g in output_batch["target"]
                ]
                p_writer.writelines(s + "\n" for s in predictions)
                t_writer.writelines(s + "\n" for s in gts)
                predictions = [s if len(s) > 5 else 'NULL' for s in predictions]
                gts = [s for s in gts]
                rouge1, rouge2, rougel = calculate_rouge(predictions, gts)
                output_batch['rouge-1'], output_batch['rouge-2'], output_batch['rouge-l'] = \
                    torch.tensor(rouge1), torch.tensor(rouge2), torch.tensor(rougel)
            p_writer.close()
            t_writer.close()

        return self._validation_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def _test_end(self, outputs):
        return self._validation_end(outputs)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = SummarizationDataset(self.tokenizer, type_path=type_path, **self.dataset_kwargs)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=shuffle)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        dataset_total_len = len(dataloader.dataset)
        t_total = (
            (dataset_total_len // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        if self.hparams.scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total,
                last_epoch=self.hparams.resume_from_step,
            )
        elif self.hparams.scheduler_type == 'constant':
            scheduler = get_constant_schedule_with_warmup(
                self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total,
                last_epoch=self.hparams.resume_from_step,
            )
        elif self.hparams.scheduler_type == 'inverse_sqrt':
            scheduler = get_inverse_sqrt_schedule_with_warmup(
                self.opt, num_warmup_steps=self.hparams.warmup_steps,
                last_epoch=self.hparams.resume_from_step,
            )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        # Add BART specific options
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--dataset_name",
            default="cnn_dm",
            type=str,
            help="The data to train/evaluate on.",
        )
        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.",
        )
        parser.add_argument(
            "--run_id",
            default='00',
            type=str,
            help="",
        )
        parser.add_argument(
            "--train_from_scratch",
            action="store_true",
            help="Reinitialize all model weights.",
        )
        parser.add_argument(
            "--overwrite_output_dir",
            action="store_true",
            help="",
        )
        parser.add_argument(
            "--label_smooth",
            default=0.,
            type=float,
        )
        parser.add_argument(
            "--role_regularization",
            default=0.,
            type=float,
            help="The role regularization coefficient"
        )
        parser.add_argument(
            "--token_regularization",
            default=0.,
            type=float,
            help="The role regularization coefficient"
        )
        parser.add_argument(
            "--resume_from_epoch",
            default=0,
            type=int,
            help="The checkpoint to restore from."
        )
        parser.add_argument(
            "--resume_from_step",
            default=-1,
            type=int,
            help="The checkpoint to restore from."
        )
        parser.add_argument(
            "--resume_ckpt_path",
            default=None,
            type=str,
        )
        parser.add_argument(
            "--num_roles_per_layer",
            default=20,
            type=int,
            help="The number of role vectors in tp-transformer."
        )
        parser.add_argument(
            "--dont_prepend_bos",
            action="store_true",
            help=""
        )
        parser.add_argument(
            "--transfer_from_dataset",
            type=str,
            default=None,
        )
        return parser


def main(args):
    args.data_dir = os.path.join(args.data_dir, args.dataset_name)
    args.output_dir = os.path.join(args.output_dir, args.dataset_name, args.run_id)
    if args.resume_from_epoch > 0:
        if args.transfer_from_dataset is not None:
            args.resume_ckpt_path = os.path.join(args.resume_ckpt_path, args.transfer_from_dataset, args.run_id)
        else:
            args.resume_ckpt_path = os.path.join(args.resume_ckpt_path, args.dataset_name, args.run_id)

        checkpoints = list(sorted(
            glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"),
                      recursive=True), key=os.path.getmtime))
        if len(checkpoints) == 0:
            checkpoints = list(sorted(
                glob.glob(os.path.join(args.resume_ckpt_path, "checkpointepoch=" + str(args.resume_from_epoch) + ".ckpt"),
                          recursive=True)))
        args.resume_ckpt_path = checkpoints[-1]
        args.resume_from_step = torch.load(args.resume_ckpt_path)["global_step"]
        print("Resuming from checkpoint:")
        print(args.resume_ckpt_path)
        print("Resuming from step:")
        print(args.resume_from_step)
    else:
        checkpoints = list(sorted(
            glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"),
                      recursive=True), key=os.path.getmtime))
        if len(checkpoints) > 0:
            args.resume_ckpt_path = checkpoints[-1]
            args.resume_from_step = torch.load(args.resume_ckpt_path)["global_step"]
            print("Resuming from checkpoint:")
            print(args.resume_ckpt_path)
            print("Resuming from step:")
            print(args.resume_from_step)

    # If output_dir not provided, a folder will be generated in pwd
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = SummarizationTrainer(args)
    trainer = generic_train(model, args)

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        # See https://github.com/huggingface/transformers/issues/3159
        # pl use this format to create a checkpoint:
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master\
        # /pytorch_lightning/callbacks/model_checkpoint.py#L169
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = SummarizationTrainer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    main(args)