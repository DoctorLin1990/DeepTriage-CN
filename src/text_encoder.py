#!/usr/bin/env python3
"""
text_encoder.py

Frozen BERT-Chinese encoder for extracting 768-dimensional [CLS] embeddings
from nurse-recorded chief complaint text, as described in Section 3.5 of the
paper.

Design decisions (from paper):
    - Model : bert-base-chinese (110M parameters)
    - Frozen : torch.no_grad() — weights are never updated
    - Max tokens : 64 (paper Section 3.4; covers >99th percentile of complaint
                   lengths given mean of 8.3 words ≈ 10–15 tokens)
    - Representation : final hidden state of the [CLS] token → 768-d vector

Rationale for frozen encoder (Section 3.4):
    Fine-tuning the full bert-base-chinese on a moderate-sized dataset
    (8,000 samples) resulted in severe overfitting within three epochs.
    The frozen-encoder strategy preserves pre-trained linguistic knowledge
    while mitigating overfitting and reducing computational cost.
"""

import numpy as np
import torch
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class TextEncoder:
    """
    Deterministic, frozen BERT-Chinese feature extractor.

    Extracts the 768-d [CLS] token representation for each input string.
    The BERT weights are never modified (torch.no_grad() throughout).

    Args:
        model_name : HuggingFace model identifier.
                     Default: 'bert-base-chinese' (Section 3.5).
        max_length : Maximum tokenisation length in sub-word tokens.
                     Default: 64 (Section 3.4).
        device     : Torch device string ('cpu', 'cuda', 'cuda:0', …).
                     Auto-detected if None.
        batch_size : Number of texts processed per forward pass.
                     Reduce if GPU OOM; increase for faster CPU throughput.
    """

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        max_length: int = 64,
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.max_length  = max_length
        self.batch_size  = batch_size
        self._tokenizer  = None
        self._model      = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load tokeniser and BERT model from HuggingFace Hub on first call.

        Loading is deferred until encode() is called to keep import time fast
        and avoid downloading weights unnecessarily (e.g. during unit tests
        that mock this class).
        """
        if self._tokenizer is None:
            try:
                from transformers import BertTokenizer, BertModel
            except ImportError as e:
                raise ImportError(
                    "The 'transformers' package is required for TextEncoder.  "
                    "Install it with: pip install transformers"
                ) from e

            logger.info(
                f"Loading BERT tokeniser and model '{self.model_name}' "
                f"(device={self.device}) …"
            )
            self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self._model = BertModel.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)
            logger.info("BERT model loaded and frozen (eval mode).")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings into 768-d [CLS] embeddings.

        The model runs under torch.no_grad() — no gradients are computed and
        the weights are never modified.

        Args:
            texts : List of raw chief complaint strings (after [MISSING]
                    substitution by preprocess.preprocess_text).

        Returns:
            np.ndarray of shape (n, 768), float32.
        """
        self._load_model()

        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]

            encoding = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            # Move tensors to the correct device
            input_ids      = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            token_type_ids = encoding.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                # [CLS] token is the first token; shape (batch, 768)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embeddings.cpu().numpy())

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        logger.debug(f"Encoded {len(texts)} texts → shape {embeddings.shape}")
        return embeddings
