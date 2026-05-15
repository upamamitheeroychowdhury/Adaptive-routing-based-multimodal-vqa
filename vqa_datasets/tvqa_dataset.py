import json
import pickle
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class TVQATerraDataset(Dataset):
    def __init__(
        self,
        json_path,
        vcpt_path,
        tokenizer_name="bert-base-uncased",
        # max_len=256,
        max_len=128,
        max_reason_len=128,
        reasoning_path=None,
        with_ts=True
    ):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        with open(vcpt_path, "rb") as f:
            self.vcpt_dict = pickle.load(f)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.max_reason_len = max_reason_len
        self.with_ts = with_ts

        self.reasoning = {}
        if reasoning_path is not None:
            with open(reasoning_path, "r") as f:
                self.reasoning = json.load(f)

    def __len__(self):
        return len(self.data)

    def get_sub(self, item):
        return item.get("located_sub_text", "") if self.with_ts else item.get("sub_text", "")

    def get_vcpt(self, item):
        vid = item["vid_name"]

        if vid not in self.vcpt_dict:
            return ""

        vcpt = self.vcpt_dict[vid]

        if isinstance(vcpt, list):
            vcpt = " ".join(vcpt)

        return str(vcpt)

    def __getitem__(self, idx):
        item = self.data[idx]

        q = item["q"]
        answers = [item[f"a{i}"] for i in range(5)]
        sub = self.get_sub(item)
        vcpt = self.get_vcpt(item)

        encoded_choices = []

        for ans in answers:
            text = f"question: {q} answer: {ans} subtitle: {sub} visual concepts: {vcpt}"

            enc = self.tokenizer(
                text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            encoded_choices.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)
            })

        input_ids = torch.stack([x["input_ids"] for x in encoded_choices], dim=0)
        attention_mask = torch.stack([x["attention_mask"] for x in encoded_choices], dim=0)

        label = int(item["answer_idx"])
        qid = item["qid"]

        reason_text = self.reasoning.get(str(qid), "")
        if reason_text == "":
            reason_text = f"The question is about: {q}"

        r_enc = self.tokenizer(
            reason_text,
            max_length=self.max_reason_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label).long(),
            "reasoning_input_ids": r_enc["input_ids"].squeeze(0),
            "reasoning_attention_mask": r_enc["attention_mask"].squeeze(0),
            "qid": qid
        }