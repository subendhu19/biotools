from __future__ import absolute_import, division, print_function

import logging
import os

from transformers.data.processors.utils import DataProcessor, InputExample

logger = logging.getLogger(__name__)

class SnliProcessor(DataProcessor):
    def __init__(self, task = "snli"):
        self.task = task

    def get_train_examples(self, data_dir):
        """See base class."""
        lg = self.language if self.train_language is None else self.train_language
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-MT-1.0/multinli/multinli.train.{}.tsv".format(lg)))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ('train', i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2] == "contradictory" else line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            language = line[0]
            if language != self.language:
                continue
            guid = "%s-%s" % ('test', i)
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

xnli_processors = {
    "xnli": XnliProcessor,
}

xnli_output_modes = {
    "xnli": "classification",
}

xnli_tasks_num_labels = {
    "xnli": 3,
}