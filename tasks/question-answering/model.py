from argparse import Namespace
from typing import Dict, Any, List
import torch
import torch.nn as nn
from models.albert import AlbertModel
from metrics import single_label_accuracy


class QAAlbert(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.encoder = AlbertModel.from_pretrained()
        self.classifier = nn.Linear(self.encoder.d_model, 2)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
        segments: torch.Tensor,
        start_labels: torch.Tensor,
        end_labels: torch.Tensor,
    ) -> Dict[str, Any]:

        hs = self.encoder(tokens, masks, segments)
        logits = self.classifier(hs)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]

        loss = (
            self.loss_fn(start_logits, start_labels)
            + self.loss_fn(end_logits, end_labels)
        ) / 2.0
        stats = {
            "loss": loss.item(),
            "acc": single_label_accuracy(
                torch.concat([start_logits.argmax(-1), end_logits.argmax(-1)]),
                torch.concat([start_labels, end_labels]),
            ),
        }

        return {"loss": loss, "stats": stats}

    def answer(
        self,
        tokens: torch.Tensor = None,
        masks: torch.Tensor = None,
        segments: torch.Tensor = None,
        offset_mapping: torch.Tensor = None,
        beam_size: int = 5,
        threshold: float = 0.0,
        contexts: List[str] = None,
        ids: List[str] = None,
    ) -> List[Dict[str, Any]]:

        hs = self.encoder(tokens, masks, segments)
        logits = self.classifier(hs)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]

        # beam search
        all_preds = []
        for batch_idx in range(tokens.size(0)):
            start_logit = start_logits[batch_idx]
            end_logit = end_logits[batch_idx]
            null_pred = {
                "prediction_text": "",
                "score": start_logit[0] + end_logit[0],
                "start_logit": start_logits[0],
                "end_logit": end_logits[0],
                "id": ids[batch_idx] if ids is not None else None,
            }
            start_preds = torch.argsort(start_logit, descending=True)[:beam_size]
            end_preds = torch.argsort(end_logit, descending=True)[:beam_size]
            valid_preds = []
            for start_idx in start_preds:
                for end_idx in end_preds:
                    if (
                        start_idx >= len(offset_mapping[batch_idx])
                        or end_idx >= len(offset_mapping[batch_idx])
                        or end_idx < start_idx
                    ):
                        continue
                    valid_preds.append(
                        {
                            "offsets": (
                                offset_mapping[batch_idx][start_idx][0],
                                offset_mapping[batch_idx][end_idx][1],
                            ),
                            "score": start_logit[start_idx] + end_logit[end_idx],
                            "start_logit": start_logit[start_idx],
                            "end_logit": end_logit[end_idx],
                            "id": ids[batch_idx] if ids is not None else None,
                        }
                    )

            # get best valid prediction
            valid_preds = sorted(valid_preds, key=lambda x: x["score"], reverse=True)
            for pred in valid_preds:
                pred["prediction_text"] = contexts[batch_idx][
                    pred["offsets"][0] : pred["offsets"][1]
                ]
                if pred["prediction_text"] != "":
                    best_valid_pred = pred
                    break

            # compare null and valid scores
            if best_valid_pred["score"] > null_pred["score"] + threshold:
                all_preds.append(best_valid_pred)
            else:
                all_preds.append(null_pred)

        return all_preds
