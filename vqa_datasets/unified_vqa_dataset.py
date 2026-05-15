import os
import json
import pickle
from collections import Counter

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


def load_json_or_jsonl(path):
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.values())

    return data


def normalize_object_name(x):
    return str(x).strip().lower().replace(" ", "_")


def extract_objects_from_entry(entry):
    """
    Handles formats like:
    ["person", "chair"]
    "person chair cup"
    {"labels": [...]}
    [{"label": "person", "conf": 0.9}, ...]
    """

    objs = []

    if entry is None:
        return objs

    if isinstance(entry, str):
        for x in entry.replace(",", " ").split():
            objs.append(normalize_object_name(x))
        return objs

    if isinstance(entry, list):
        for v in entry:
            if isinstance(v, str):
                objs.append(normalize_object_name(v))
            elif isinstance(v, dict):
                label = v.get("label") or v.get("class") or v.get("name")
                if label is not None:
                    objs.append(normalize_object_name(label))
        return objs

    if isinstance(entry, dict):
        if "labels" in entry:
            return extract_objects_from_entry(entry["labels"])
        if "objects" in entry:
            return extract_objects_from_entry(entry["objects"])
        if "detections" in entry:
            return extract_objects_from_entry(entry["detections"])

        for v in entry.values():
            objs.extend(extract_objects_from_entry(v))

    return objs


def build_object_vocab(vcpt_pkl_paths, min_freq=1, max_objects=5000):
    counter = Counter()

    for path in vcpt_pkl_paths:
        if path is None or not os.path.exists(path):
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)

        for _, video_vcpt in data.items():
            objs = extract_objects_from_entry(video_vcpt)
            counter.update(objs)

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
    }

    for obj, freq in counter.most_common(max_objects):
        if freq >= min_freq and obj not in vocab:
            vocab[obj] = len(vocab)

    return vocab


class UnifiedVQADataset(Dataset):
    def __init__(
        self,
        json_path,
        siglip_pkl_path=None,
        vcpt_pkl_path=None,
        object_vocab=None,
        tokenizer_name="bert-base-uncased",
        max_qa_len=96,
        max_sub_len=256,
        max_frames=32,
        max_objects_per_frame=10,
        siglip_dim=1152,
    ):
        self.data = load_json_or_jsonl(json_path)

        self.siglip_pkl_path = siglip_pkl_path
        self.vcpt_pkl_path = vcpt_pkl_path

        self.siglip_feats = None
        self.vcpt_feats = None

        if siglip_pkl_path is not None and os.path.exists(siglip_pkl_path):
            with open(siglip_pkl_path, "rb") as f:
                self.siglip_feats = pickle.load(f)

        if vcpt_pkl_path is not None and os.path.exists(vcpt_pkl_path):
            with open(vcpt_pkl_path, "rb") as f:
                self.vcpt_feats = pickle.load(f)

        if object_vocab is None:
            object_vocab = build_object_vocab([vcpt_pkl_path])

        self.object_vocab = object_vocab

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

        self.max_qa_len = max_qa_len
        self.max_sub_len = max_sub_len
        self.max_frames = max_frames
        self.max_objects_per_frame = max_objects_per_frame
        self.siglip_dim = siglip_dim

    def __len__(self):
        return len(self.data)

    def _get_value(self, item, keys, default=""):
        for k in keys:
            if k in item and item[k] is not None:
                return item[k]
        return default

    def _get_answers(self, item):
        if "answers" in item:
            answers = item["answers"]
        else:
            answers = [
                item.get("a0", ""),
                item.get("a1", ""),
                item.get("a2", ""),
                item.get("a3", ""),
                item.get("a4", ""),
            ]

        if len(answers) < 5:
            answers = answers + [""] * (5 - len(answers))

        return answers[:5]

    def _get_label(self, item):
        for key in ["answer_idx", "label", "answer", "gt"]:
            if key in item:
                return int(item[key])
        return -1

    def _get_video_key(self, item):
        for key in ["vid_name", "video_id", "video", "clip_name"]:
            if key in item:
                return str(item[key]).replace(".mp4", "")
        return ""

    def _tokenize(self, text, max_len):
        enc = self.tokenizer(
            str(text),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def _lookup_feature(self, feat_dict, video_key):
        if feat_dict is None:
            return None

        candidates = [
            video_key,
            video_key + ".mp4",
            video_key.replace(".mp4", ""),
        ]

        for k in candidates:
            if k in feat_dict:
                return feat_dict[k]

        return None

    def _load_siglip(self, item):
        video_key = self._get_video_key(item)

        feat = self._lookup_feature(self.siglip_feats, video_key)

        if feat is None and "siglip_path" in item:
            feat = torch.load(item["siglip_path"], map_location="cpu")

        if feat is None:
            feat = torch.zeros(self.max_frames, self.siglip_dim)

        if isinstance(feat, dict):
            for key in ["features", "siglip", "feat", "video_feats"]:
                if key in feat:
                    feat = feat[key]
                    break

        feat = torch.tensor(feat).float()

        if feat.dim() == 1:
            feat = feat.unsqueeze(0)

        T, D = feat.shape

        if D != self.siglip_dim:
            raise ValueError(
                f"SigLIP dim mismatch for {video_key}: expected {self.siglip_dim}, got {D}"
            )

        if T >= self.max_frames:
            feat = feat[: self.max_frames]
            mask = torch.ones(self.max_frames, dtype=torch.long)
        else:
            pad = torch.zeros(self.max_frames - T, self.siglip_dim)
            feat = torch.cat([feat, pad], dim=0)

            mask = torch.cat(
                [
                    torch.ones(T, dtype=torch.long),
                    torch.zeros(self.max_frames - T, dtype=torch.long),
                ],
                dim=0,
            )

        return feat, mask

    def _vcpt_to_frames(self, raw_vcpt):
        """
        Converts raw VCPT into frame-level object lists:
        [
          ["person", "chair"],
          ["person", "table"],
          ...
        ]
        """

        if raw_vcpt is None:
            return []

        if isinstance(raw_vcpt, dict):
            frame_keys = list(raw_vcpt.keys())

            # frame-level dict
            if len(frame_keys) > 0:
                frame_keys = sorted(frame_keys, key=lambda x: str(x))
                frames = []

                for k in frame_keys:
                    objs = extract_objects_from_entry(raw_vcpt[k])
                    frames.append(objs)

                return frames

        if isinstance(raw_vcpt, list):
            # list of frames
            if len(raw_vcpt) > 0 and isinstance(raw_vcpt[0], (list, dict)):
                return [extract_objects_from_entry(x) for x in raw_vcpt]

            # aggregated object list
            return [extract_objects_from_entry(raw_vcpt)]

        if isinstance(raw_vcpt, str):
            return [extract_objects_from_entry(raw_vcpt)]

        return []

    def _load_vcpt_objects(self, item):
        video_key = self._get_video_key(item)

        raw_vcpt = self._lookup_feature(self.vcpt_feats, video_key)

        frames = self._vcpt_to_frames(raw_vcpt)

        obj_ids = torch.zeros(
            self.max_frames,
            self.max_objects_per_frame,
            dtype=torch.long,
        )

        frame_mask = torch.zeros(self.max_frames, dtype=torch.long)

        if len(frames) == 0:
            return obj_ids, frame_mask

        if len(frames) >= self.max_frames:
            frames = frames[: self.max_frames]

        for t, objs in enumerate(frames):
            if len(objs) == 0:
                continue

            frame_mask[t] = 1

            objs = objs[: self.max_objects_per_frame]

            for j, obj in enumerate(objs):
                obj_ids[t, j] = self.object_vocab.get(obj, self.object_vocab["<unk>"])

        return obj_ids, frame_mask

    def __getitem__(self, idx):
        item = self.data[idx]

        question = self._get_value(item, ["q", "question"], "")
        answers = self._get_answers(item)

        qa_ids = []
        qa_masks = []

        for ans in answers:
            text = question + " [SEP] " + str(ans)
            ids, mask = self._tokenize(text, self.max_qa_len)
            qa_ids.append(ids)
            qa_masks.append(mask)

        qa_input_ids = torch.stack(qa_ids, dim=0)
        qa_attention_mask = torch.stack(qa_masks, dim=0)

        subtitle = self._get_value(
            item,
            ["sub_text", "subtitle", "sub", "subs", "transcript", "located_sub_text"],
            "",
        )

        sub_input_ids, sub_attention_mask = self._tokenize(
            subtitle,
            self.max_sub_len,
        )

        video_feats, video_mask = self._load_siglip(item)

        vcpt_obj_ids, vcpt_obj_mask = self._load_vcpt_objects(item)

        label = self._get_label(item)

        return {
            "qa_input_ids": qa_input_ids,
            "qa_attention_mask": qa_attention_mask,

            "sub_input_ids": sub_input_ids,
            "sub_attention_mask": sub_attention_mask,

            "video_feats": video_feats,
            "video_mask": video_mask,

            "vcpt_obj_ids": vcpt_obj_ids,
            "vcpt_obj_mask": vcpt_obj_mask,

            "label": torch.tensor(label, dtype=torch.long),

            "qid": str(self._get_value(item, ["qid", "id"], str(idx))),
            "vid_name": self._get_video_key(item),
        }