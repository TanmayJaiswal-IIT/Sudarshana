"""Token classification head module for Sudarshana indirect prompt-injection detection.

Features:
- DeBERTa-v3 encoder backbone.
- 3-class token classification head (O: 0, B-INJ: 1, I-INJ: 2).
- Class-weighted CrossEntropyLoss to address severe class imbalance (O vs B-INJ/I-INJ).
- Joint-loss hook (L_detect + lambda_tok * L_tok) for P3 integration.
- Standalone forward pass tested on arbitrary tensors.
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel


class TokenClassificationHead(nn.Module):
    """
    Token-level classification projection head.
    Projects encoder hidden states -> logits for (O, B-INJ, I-INJ).
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 3,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            logits: (batch_size, seq_len, num_labels)
        """
        x = self.dropout(hidden_states)
        logits = self.classifier(x)
        return logits


class SudarshanaTokenClassifier(nn.Module):
    """
    Prototype token classifier built on a shared DeBERTa-v3 encoder.
    P3 can integrate this head into the joint model.
    """
    def __init__(
        self,
        backbone_name_or_path: str = "microsoft/deberta-v3-xsmall",
        num_labels: int = 3,
        dropout_prob: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(backbone_name_or_path)
        self.encoder = AutoModel.from_pretrained(backbone_name_or_path, config=self.config).float()
        self.head = TokenClassificationHead(
            hidden_size=self.config.hidden_size,
            num_labels=num_labels,
            dropout_prob=dropout_prob
        ).float()

        # Default class weights: penalize missed injection tokens (B-INJ and I-INJ) heavily
        # Class 0 (O): 1.0, Class 1 (B-INJ): 8.0, Class 2 (I-INJ): 8.0
        if class_weights is None:
            self.register_buffer("class_weights", torch.tensor([1.0, 8.0, 8.0], dtype=torch.float))
        else:
            self.register_buffer("class_weights", class_weights.float())

        self.loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=-100
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) with values in {0, 1, 2, -100}
        Returns:
            dict containing 'logits', 'loss' (if labels provided), 'hidden_states'
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state.float()  # (B, L, H)
        logits = self.head(hidden_states)                          # (B, L, 3)

        loss = None
        if labels is not None:
            # Flatten tokens for CrossEntropyLoss
            active_logits = logits.view(-1, self.num_labels)
            active_labels = labels.view(-1)
            loss = self.loss_fct(active_logits, active_labels)

        if not return_dict:
            return (loss, logits) if loss is not None else logits

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states
        }

    @staticmethod
    def compute_joint_loss(
        loss_detect: torch.Tensor,
        loss_tok: torch.Tensor,
        lambda_tok: float = 1.0
    ) -> torch.Tensor:
        """
        Hook for P3 joint training run: L = L_detect + lambda_tok * L_tok.
        P3 tunes lambda_tok in the joint run.
        """
        return loss_detect + lambda_tok * loss_tok
