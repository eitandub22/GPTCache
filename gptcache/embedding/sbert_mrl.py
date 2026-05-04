import numpy as np
from gptcache.utils import import_sbert
from gptcache.embedding.base import BaseEmbedding

import_sbert()

from sentence_transformers import SentenceTransformer  # pylint: disable=C0413


class SBERTMRL(BaseEmbedding):
    """Generate truncated sentence embeddings using Matryoshka Representation Learning (MRL).

    MRL-trained models pack the most critical semantic information into the
    first dimensions of the embedding vector. This class loads an MRL-compatible
    model, truncates the output to ``target_dim``, and L2-normalizes the result.

    The multiplicative benefit: dimension reduction (e.g. 768→256 = 3x) stacks
    with downstream quantization (SQ8 = 4x) for ~12x total compression.

    :param model: MRL-compatible model name, defaults to 'nomic-ai/nomic-embed-text-v1.5'.
    :type model: str
    :param target_dim: target dimensionality after truncation, defaults to 256.
    :type target_dim: int
    :param trust_remote_code: whether to trust remote code for model loading, defaults to True.
    :type trust_remote_code: bool

    Example:
        .. code-block:: python

            from gptcache.embedding import SBERTMRL

            test_sentence = 'Hello, world.'
            encoder = SBERTMRL('nomic-ai/nomic-embed-text-v1.5', target_dim=256)
            embed = encoder.to_embeddings(test_sentence)
            assert len(embed) == 256
    """

    def __init__(
        self,
        model: str = "nomic-ai/nomic-embed-text-v1.5",
        target_dim: int = 256,
        trust_remote_code: bool = True,
    ):
        self.model = SentenceTransformer(model, trust_remote_code=trust_remote_code)
        self.model.eval()
        self._target_dim = target_dim

        # Validate that target_dim doesn't exceed the model's native dimension
        full_dim = self.model.get_embedding_dimension() 
        if target_dim > full_dim:
            raise ValueError(
                f"target_dim={target_dim} exceeds model's native dimension={full_dim}. "
                f"MRL truncation can only reduce dimensions, not increase them."
            )

    def to_embeddings(self, data, **_):
        """Generate MRL-truncated embedding given text input.

        :param data: text in string.
        :type data: str

        :return: a truncated, L2-normalized embedding in shape of (target_dim,).
        """
        if not isinstance(data, list):
            data = [data]
        emb = self.model.encode(data)

        # MRL truncation: slice to target dimension
        truncated = emb[:, :self._target_dim]

        # L2-normalize after truncation (critical for cosine similarity)
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        normalized = truncated / np.maximum(norms, 1e-9)

        return np.array(normalized.squeeze(0)).astype("float32")

    @property
    def dimension(self):
        """Embedding dimension (after MRL truncation).

        :return: target dimension
        """
        return self._target_dim
