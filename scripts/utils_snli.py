from __future__ import absolute_import, division, print_function

import logging
import os
from os.path import join
import jsonlines
from transformers.data.processors.utils import DataProcessor, InputExample

logger = logging.getLogger(__name__)

class SnliProcessor(DataProcessor):
    def __init__(self, task = "snli"):
        self.task = task

    def read_data(self, file_name):
        fp = open(file_name, 'r')
        reader = jsonlines.Reader(fp)
        curr_data = []
        for obj in reader:
            curr_data.append(obj)
        reader.close()
        fp.close()
        return curr_data

    def get_train_examples(self, data_dir):
        """See base class."""
        # lg = self.language if self.train_language is None else self.train_language
        # lines = self._read_tsv(os.path.join(data_dir, "XNLI-MT-1.0/multinli/multinli.train.{}.tsv".format(lg)))
        train_data = read_data(join(data_dir, 'snli_1.0_train.jsonl'))
        dev_data = read_data(join(data_dir, 'snli_1.0_dev.jsonl'))
        lines = train_data+dev_data
        lines = [x for x in lines if(x['gold_label']!='-')]
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ('train', i)
            text_a = line['sentence1']
            text_b = line['sentence2']
            # label = "contradiction" if line[2] == "contradictory" else line[2]
            label = line['gold_label']
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        # lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
        lines = read_data(join(data_dir, 'snli_1.0_test.jsonl'))
        lines = [x for x in lines if (x['gold_label'] != '-')]
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            language = line[0]
            if language != self.language:
                continue
            guid = "%s-%s" % ('test', i)
            text_a = line['sentence1']
            text_b = line['sentence2']
            label = line['gold_label']
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

# xnli_processors = {
#     "xnli": XnliProcessor,
# }

xnli_output_modes = {
    "xnli": "classification",
}

xnli_tasks_num_labels = {
    "xnli": 3,
}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def xnli_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "xnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)