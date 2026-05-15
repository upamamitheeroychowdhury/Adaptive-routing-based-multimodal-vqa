import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class TERRA_VQA(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout=0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.answer_scorer = nn.Linear(hidden, 1)

    def encode(self, input_ids, attention_mask):
        bsz, num_choices, seq_len = input_ids.size()

        flat_ids = input_ids.view(bsz * num_choices, seq_len)
        flat_mask = attention_mask.view(bsz * num_choices, seq_len)

        out = self.bert(
            input_ids=flat_ids,
            attention_mask=flat_mask
        )

        cls = out.last_hidden_state[:, 0]  # [B*5, H]
        cls = self.dropout(cls)

        scores = self.answer_scorer(cls).view(bsz, num_choices)
        reps = cls.view(bsz, num_choices, -1)

        return scores, reps

    def forward(self, input_ids, attention_mask, labels=None,
                reasoning_input_ids=None, reasoning_attention_mask=None,
                lambda_reason=0.1):

        scores, reps = self.encode(input_ids, attention_mask)

        loss = None
        answer_loss = None
        reason_loss = None

        if labels is not None:
            answer_loss = F.cross_entropy(scores, labels)
            loss = answer_loss

        if reasoning_input_ids is not None and reasoning_attention_mask is not None and labels is not None:
            r_out = self.bert(
                input_ids=reasoning_input_ids,
                attention_mask=reasoning_attention_mask
            )
            r_cls = r_out.last_hidden_state[:, 0]  # [B, H]

            correct_rep = reps[torch.arange(reps.size(0), device=reps.device), labels]
            reason_loss = 1.0 - F.cosine_similarity(correct_rep, r_cls, dim=-1).mean()

            loss = loss + lambda_reason * reason_loss

        return {
            "loss": loss,
            "answer_loss": answer_loss,
            "reason_loss": reason_loss,
            "scores": scores
        }