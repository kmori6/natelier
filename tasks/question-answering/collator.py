from argparse import Namespace
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer


class QABatchCollator:
    def __init__(self, args: Namespace, return_references: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.max_length = min(args.max_length, self.tokenizer.model_max_length)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.return_references = return_references

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        encodings = self.tokenizer(
            [data["question"].lstrip().rstrip() for data in batch],
            [data["context"] for data in batch],
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        start_labels, end_labels = [], []
        for batch_idx, offset_mapping in enumerate(encodings["offset_mapping"]):
            # no answer: cls token label
            if len(batch[batch_idx]["answers"]["answer_start"]) == 0:
                start_labels.append(self.cls_token_id)
                end_labels.append(self.cls_token_id)
            else:
                answer_start = batch[batch_idx]["answers"]["answer_start"][0]
                answer_end = answer_start + len(batch[batch_idx]["answers"]["text"][0])
                context_indices = [
                    idx
                    for idx, sequence_id in enumerate(encodings.sequence_ids(batch_idx))
                    if sequence_id == 1
                ]
                # out-of-context answer: cls token label
                if (
                    answer_start < offset_mapping[context_indices[0], 0]
                    or offset_mapping[context_indices[-1], 1] < answer_end
                ):
                    start_labels.append(self.cls_token_id)
                    end_labels.append(self.cls_token_id)
                # valid start and end labels
                else:
                    for token_idx in context_indices:
                        if offset_mapping[:, 0][token_idx + 1] == answer_start:
                            start_label = token_idx + 1
                            break
                        if offset_mapping[:, 0][token_idx + 1] > answer_start:
                            start_label = token_idx
                            break
                    for idx in reversed(context_indices):
                        if offset_mapping[:, 1][idx - 1] == answer_end:
                            end_label = idx - 1
                            break
                        if offset_mapping[:, 1][idx - 1] < answer_end:
                            end_label = idx
                            break
                    start_labels.append(start_label)
                    end_labels.append(end_label)
        encodings.update(
            start_labels=torch.tensor(start_labels, dtype=torch.long),
            end_labels=torch.tensor(end_labels, dtype=torch.long),
        )
        if self.return_references:
            encodings.update(
                references=[
                    {"answers": data["answers"], "id": data["id"]} for data in batch
                ],
                contexts=[data["context"] for data in batch],
                ids=[data["id"] for data in batch],
            )
            return {k: v for k, v in encodings.items()}
        else:
            return {k: v for k, v in encodings.items() if k != "offset_mapping"}
