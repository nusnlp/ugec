import logging
import os
from dataclasses import dataclass, field
from typing import Optional
import itertools
import torch
from omegaconf import II

from fairseq.data import (
    Dictionary,
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    MaskTokensDataset,
    TruncateDataset,
    indexed_dataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, TranslationConfig
from fairseq import utils

logger = logging.getLogger(__name__)


def load_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        prepend_bos_src=None,
        mask_prob=0,
        mask_whole_words=False,
        mask_idx=None,
        seed=1,
        leave_unmasked_prob=0.1,
        random_token_prob=0.1,
        freq_weighted_replacement=False,
        mask_multiple_length=1,
        mask_stdev=0.0,
        args=None,
        train_label_path="None",
        valid_label_path="None",

):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(
            data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    label = None
    if split == 'train':
        if train_label_path != "None":
            f = open(train_label_path).readlines()
            label = [int(i.strip()) for i in f]
            label = torch.tensor(label)
    elif split == 'valid':
        if valid_label_path != "None":
            f = open(valid_label_path).readlines()
            label = [int(i.strip()) for i in f]
            label = torch.tensor(label)
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(
                data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(
                data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )

        if mask_prob > 0:
            mask_whole_words = (
                get_whole_word_mask(args, src_dict)
                if mask_whole_words
                else None
            )
            src_dataset, _ = MaskTokensDataset.apply_mask(
                src_dataset,
                src_dict,
                pad_idx=src_dict.pad(),
                mask_idx=mask_idx,
                seed=seed,
                mask_prob=mask_prob,
                leave_unmasked_prob=leave_unmasked_prob,
                random_token_prob=random_token_prob,
                freq_weighted_replacement=freq_weighted_replacement,
                mask_whole_words=mask_whole_words,
                mask_multiple_length=mask_multiple_length,
                mask_stdev=mask_stdev,
            )

        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )

        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(
            tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(
            data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        label=label,
    )


@dataclass
class GrammarCorrectionConfig(TranslationConfig):
    train_label: str = field(
        default="None",
        metadata={"help": "path to the train label file"},
    )
    valid_label: str = field(
        default="None",
        metadata={"help": "path to the validation label file"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={
            "help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    seed: int = II("common.seed")


@register_task("grammar_correction", dataclass=GrammarCorrectionConfig)
class GrammarCorrectionTask(TranslationTask):
    cfg: GrammarCorrectionConfig

    def __init__(self, cfg: GrammarCorrectionConfig, src_dict: Dictionary, tgt_dict: Dictionary):
        super().__init__(cfg, src_dict, tgt_dict)
        self.mask_idx = self.src_dict.add_symbol("<mask>")
        self.mask_idx = self.tgt_dict.add_symbol("<mask>")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            mask_prob=self.cfg.mask_prob,
            mask_whole_words=self.cfg.mask_whole_words,
            mask_idx=self.mask_idx,
            seed=self.cfg.seed,
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
            freq_weighted_replacement=self.cfg.freq_weighted_replacement,
            mask_multiple_length=self.cfg.mask_multiple_length,
            mask_stdev=self.cfg.mask_stdev,
            args=self.cfg,
            train_label_path=self.cfg.train_label,
            valid_label_path=self.cfg.valid_label
        )
