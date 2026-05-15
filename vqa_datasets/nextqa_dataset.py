import json
import torch
from torch.utils.data.dataset import Dataset

from utils import load_pickle


class NextQADataset(Dataset):
    def __init__(self, json_path, word2idx_path, with_ts=True):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.word2idx = load_pickle(word2idx_path)
        self.with_ts = with_ts

        if self.with_ts:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "located_sub_text"]
        else:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub_text"]

    def __len__(self):
        return len(self.data)

    @classmethod
    def line_to_words(cls, line, eos=True, downcase=True):
        eos_word = "<eos>"
        words = line.lower().split() if downcase else line.split()
        words = [w for w in words if w != ","]
        words = words + [eos_word] if eos else words
        return words

    def numericalize(self, sentence, eos=True):
        return [
            self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
            for w in self.line_to_words(sentence, eos=eos)
        ]

    def __getitem__(self, index):
        item = self.data[index]
        items = []

        for k in self.text_keys:
            items.append(self.numericalize(item.get(k, "")))

        # dummy vcpt
        items.append(self.numericalize(""))

        # label
        items.append(int(item["answer_idx"]))

        # qid
        items.append(item["qid"])

        # vid_name
        items.append(item["vid_name"])

        # dummy video tensor
        cur_vid_feat = torch.zeros([2, 2]).float()
        items.append(cur_vid_feat)

        return items