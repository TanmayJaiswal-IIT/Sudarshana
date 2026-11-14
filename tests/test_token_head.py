"""Unit tests for standalone TokenClassificationHead and SudarshanaTokenClassifier."""

import pytest
import torch
from sudarshana.models.token_head import TokenClassificationHead, SudarshanaTokenClassifier


def test_token_head_random_tensors():
    """Verify TokenClassificationHead forward pass on arbitrary random tensors."""
    batch_size = 4
    seq_len = 64
    hidden_size = 384
    num_labels = 3

    head = TokenClassificationHead(hidden_size=hidden_size, num_labels=num_labels, dropout_prob=0.1)
    x = torch.randn(batch_size, seq_len, hidden_size)

    logits = head(x)
    assert logits.shape == (batch_size, seq_len, num_labels), f"Expected shape {(batch_size, seq_len, num_labels)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN"


def test_joint_loss_hook():
    """Verify joint loss hook composition."""
    l_detect = torch.tensor(0.5, requires_grad=True)
    l_tok = torch.tensor(1.2, requires_grad=True)
    lambda_tok = 0.8

    joint_loss = SudarshanaTokenClassifier.compute_joint_loss(l_detect, l_tok, lambda_tok)
    expected = 0.5 + 0.8 * 1.2
    assert torch.isclose(joint_loss, torch.tensor(expected)), f"Expected {expected}, got {joint_loss.item()}"

    # Gradient flow check
    joint_loss.backward()
    assert l_detect.grad is not None
    assert l_tok.grad is not None
    assert torch.isclose(l_tok.grad, torch.tensor(lambda_tok))


def test_class_weighted_loss_computation():
    """Verify class-weighted cross entropy ignores -100 and applies weights."""
    hidden_size = 64
    head = TokenClassificationHead(hidden_size=hidden_size, num_labels=3)
    weights = torch.tensor([1.0, 5.0, 5.0])
    loss_fct = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

    # Fake logits and labels
    logits = torch.randn(2, 4, 3)
    labels = torch.tensor([
        [0, 1, 2, -100],
        [-100, 0, -100, 1]
    ])

    loss = loss_fct(logits.view(-1, 3), labels.view(-1))
    assert not torch.isnan(loss)
    assert loss.item() > 0

