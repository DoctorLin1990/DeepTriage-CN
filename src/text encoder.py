#!/usr/bin/env python3
"""
text_encoder.py

Frozen BERT-Chinese encoder for chief complaint text, as described in
Section 3.4 and Section 3.5 of the paper.

Key design choices:
    - Model: bert-base-chinese (110M parameters)
    - Frozen weights (torch.no_grad()) to prevent overfitting on moderate
      sample size (8,000 training visits)
    - Max sequence length: 64 tokens (covers >99th percentile of complaint
      lengths, which average 8.3 words ≈ 10-15 tokens)
    - Output: 768-dimensional [CLS] embedding, serving as a document-level
      representation
    - Special token [MISSING] is used for empty chief complaint fields (2.4%
      of encounters)

Usage:
    encoder = TextEncoder(model_name="bert-base-chinese")
    embeddings = encoder.encode(list_of_texts)
"""

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class TextEncoder:
    """
    Frozen BERT-Chinese feature extractor for triage chief complaint text.

    Attributes:
        model_name: HuggingFace model identifier.
        max_length: Maximum token length for input sequences.
        device: Device on which to run the model (cuda if available, else cpu).
        tokenizer: Pretrained BertTokenizer.
        model: Pretrained BertModel with frozen parameters.
    """

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        max_length: int = 64,
        device: str = None
    ):
        """
        Initialize the text encoder.

        Args:
            model_name: HuggingFace model identifier.
            max_length: Maximum sequence length (Section 3.4: 64).
            device: Computation device. Defaults to CUDA if available.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters (Section 3.5)
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(f"TextEncoder initialized. Frozen BERT on {self.device}")

    def encode(
        self,
        texts: Union[List[str], np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode a list of chief complaint strings into BERT [CLS] embeddings.

        Args:
            texts: List of chief complaint strings in Chinese.
            batch_size: Batch size for inference.

        Returns:
            np.ndarray of shape (n_texts, 768) containing [CLS] embeddings.
        """
        texts = list(texts)
        all_embeddings = []

        with torch.no_grad():  # Disable gradient computation (frozen encoder)
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # Forward pass
                outputs = self.model(**encoded)

                # Extract [CLS] token embeddings (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)

        embeddings = np.vstack(all_embeddings)
        logger.debug(f"Encoded {len(texts)} texts into shape {embeddings.shape}")
        return embeddings


# Convenience function for quick testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    encoder = TextEncoder()
    sample_texts = ["胸痛伴胸闷", "[MISSING]", "呼吸困难加重三天"]
    emb = encoder.encode(sample_texts)
    print(f"Embedding shape: {emb.shape}")  # Expected: (3, 768)