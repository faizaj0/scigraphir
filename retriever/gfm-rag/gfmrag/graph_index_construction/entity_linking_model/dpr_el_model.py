import hashlib
import os
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from .base_model import BaseELModel


def _dev() -> str:
    """cuda > mps > cpu. Upstream hard-coded `cuda if available else cpu`, which on a Mac put the
    BGE-large encode of every entity phrase (~50k for a 4.7k-doc corpus, run TWICE: index + link)
    and the top-k matmul on the CPU: 14 of the ~55 min of a SIR-4 split on 18 Aug. MPS runs the
    same arithmetic; only float rounding differs, so a borderline synonymy edge at the 0.9
    threshold can flip. Graphs already built keep their cached embeddings and are unaffected.
    EL_DEVICE=cpu reproduces the old path exactly."""
    forced = os.environ.get("EL_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DPRELModel(BaseELModel):
    """
    Entity Linking Model based on Dense Passage Retrieval (DPR).

    This class implements an entity linking model using DPR architecture and SentenceTransformer
    for encoding entities and computing similarity scores between mentions and candidate entities.

    Args:
        model_name (str): Name or path of the SentenceTransformer model to use
        root (str, optional): Root directory for caching embeddings. Defaults to "tmp".
        use_cache (bool, optional): Whether to cache and reuse entity embeddings. Defaults to True.
        normalize (bool, optional): Whether to L2-normalize embeddings. Defaults to True.
        batch_size (int, optional): Batch size for encoding. Defaults to 32.
        query_instruct (str, optional): Instruction/prompt prefix for query encoding. Defaults to "".
        passage_instruct (str, optional): Instruction/prompt prefix for passage encoding. Defaults to "".
        model_kwargs (dict, optional): Additional kwargs to pass to SentenceTransformer. Defaults to None.

    Methods:
        index(entity_list): Indexes a list of entities by computing and caching their embeddings
        __call__(ner_entity_list, topk): Links named entities to indexed entities and returns top-k matches

    Examples:
        >>> model = DPRELModel('sentence-transformers/all-mpnet-base-v2')
        >>> model.index(['Paris', 'London', 'Berlin'])
        >>> results = model(['paris city'], topk=2)
        >>> print(results)
        {'paris city': [{'entity': 'Paris', 'score': 0.82, 'norm_score': 1.0},
                        {'entity': 'London', 'score': 0.35, 'norm_score': 0.43}]}
    """

    def __init__(
        self,
        model_name: str,
        root: str = "tmp",
        use_cache: bool = True,
        normalize: bool = True,
        batch_size: int = 32,
        query_instruct: str = "",
        passage_instruct: str = "",
        model_kwargs: dict | None = None,
    ) -> None:
        """Initialize DPR Entity Linking Model.

        Args:
            model_name (str): Name or path of the pre-trained model to load.
            root (str, optional): Root directory for cache storage. Defaults to "tmp".
            use_cache (bool, optional): Whether to use cache for embeddings. Defaults to True.
            normalize (bool, optional): Whether to normalize the embeddings. Defaults to True.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
            query_instruct (str, optional): Instruction prefix for query encoding. Defaults to "".
            passage_instruct (str, optional): Instruction prefix for passage encoding. Defaults to "".
            model_kwargs (dict | None, optional): Additional arguments to pass to the model. Defaults to None.
        """

        self.model_name = model_name
        self.use_cache = use_cache
        self.normalize = normalize
        self.batch_size = batch_size
        self.root = os.path.join(root, f"{self.model_name.replace('/', '_')}_dpr_cache")
        if self.use_cache and not os.path.exists(self.root):
            os.makedirs(self.root)
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, model_kwargs=model_kwargs
        )
        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct

    def index(self, entity_list: list) -> None:
        """
        Index a list of entities by encoding them into embeddings and optionally caching the results.

        This method processes a list of entity strings, converting them into dense vector representations
        using a pre-trained model. To avoid redundant computation, it implements a caching mechanism
        based on the MD5 hash of the input entity list.

        Args:
            entity_list (list): A list of strings representing entities to be indexed.

        Returns:
            None

        Notes:
            - The method stores the embeddings in self.entity_embeddings
            - If caching is enabled and a cache file exists for the given entity list,
              embeddings are loaded from cache instead of being recomputed
            - Cache files are stored using the MD5 hash of the concatenated entity list as filename
            - Embeddings are computed on GPU if available, otherwise on CPU
        """
        self.entity_list = entity_list
        # Get md5 fingerprint of the whole given entity list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        cache_file = f"{self.root}/{fingerprint}.pt"
        if os.path.exists(cache_file):
            self.entity_embeddings = torch.load(
                cache_file,
                map_location=_dev(),
                weights_only=True,
            )
        else:
            self.entity_embeddings = self.model.encode(
                entity_list,
                device=_dev(),
                convert_to_tensor=True,
                show_progress_bar=True,
                prompt=self.passage_instruct,
                normalize_embeddings=self.normalize,
                batch_size=self.batch_size,
            )
            if self.use_cache:
                torch.save(self.entity_embeddings, cache_file)

    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """
        Performs entity linking by matching input entities with pre-encoded entity embeddings.

        This method takes a list of named entities (e.g., from NER), computes their embeddings,
        and finds the closest matching entities from the pre-encoded knowledge base using
        cosine similarity.

        Args:
            ner_entity_list (list): List of named entities to link
            topk (int, optional): Number of top matches to return for each entity. Defaults to 1.

        Returns:
            dict: Dictionary mapping each input entity to its linked candidates. For each candidate:
                - entity (str): The matched entity name from the knowledge base
                - score (float): Raw similarity score
                - norm_score (float): Normalized similarity score (relative to top match)
        """
        _indexed = getattr(self, "entity_list", None)
        if _indexed is not None and (ner_entity_list is _indexed or ner_entity_list == _indexed) \
                and self.query_instruct == self.passage_instruct:
            # augment_graph links the indexed phrase list against itself with identical prompts,
            # so the query-side encode is bit-for-bit the index encode: reuse it (halves the phase).
            ner_entity_embeddings = self.entity_embeddings
        else:
            ner_entity_embeddings = self.model.encode(
                ner_entity_list,
                device=_dev(),
                convert_to_tensor=True,
                prompt=self.query_instruct,
                normalize_embeddings=self.normalize,
                batch_size=self.batch_size,
            )
        # PATCH (memory): the full query x entity score matrix is O(N^2) (~183 GB for 214k train
        # entities) and OOM'd the 16 GB host. Compute top-k in row-chunks to bound peak memory to
        # chunk x N. Output is identical to the full-matrix top-k.
        topk = min(topk, self.entity_embeddings.shape[0])
        _score_chunks, _idx_chunks = [], []
        for _s in range(0, ner_entity_embeddings.shape[0], 1024):
            _sc = ner_entity_embeddings[_s:_s + 1024] @ self.entity_embeddings.T
            _ts, _ti = torch.topk(_sc, topk, dim=-1)
            _score_chunks.append(_ts)
            _idx_chunks.append(_ti)
            del _sc
        # Materialise ONCE on the host. Upstream read every score with `.item()` and indexed the
        # entity list with a 0-d tensor inside the loop: ~3 device syncs per neighbour, 5.6M for
        # 56k phrases x 100 neighbours. On CPU that was the 6.5-min "Finding similar entities"
        # phase; on MPS each sync is a GPU round-trip and the same loop ran >15 min (mir_test,
        # 8 Sep). Two transfers and a pure-Python loop give identical output in seconds.
        top_k_scores = torch.cat(_score_chunks, dim=0).cpu().tolist()
        top_k_values = torch.cat(_idx_chunks, dim=0).cpu().tolist()
        linked_entity_dict: dict[str, list] = {}
        for i in range(len(ner_entity_list)):
            sorted_score = top_k_scores[i]
            sorted_indices = top_k_values[i]
            max_score = sorted_score[0]
            linked_entity_dict[ner_entity_list[i]] = [
                {
                    "entity": self.entity_list[top_k_index],
                    "score": score,
                    "norm_score": score / max_score,
                }
                for score, top_k_index in zip(sorted_score, sorted_indices)
            ]
        return linked_entity_dict


class NVEmbedV2ELModel(DPRELModel):
    """
    A DPR-based Entity Linking model specialized for NVEmbed V2 embeddings.

    This class extends DPRELModel with specific adaptations for handling NVEmbed V2 models,
    including increased sequence length and right-side padding.

    Attributes:
        model: The underlying model with max_seq_length of 32768 and right-side padding.

    Methods:
        add_eos(input_examples): Adds EOS token to input examples.
        __call__(ner_entity_list): Processes entity list with EOS tokens before linking.

    Examples:
        >>> model = NVEmbedV2ELModel('nvidia/NV-Embed-v2', query_instruct=\"Instruct: Given a entity, retrieve entities that are semantically equivalent to the given entity\\nQuery: \")
        >>> model.index(['Paris', 'London', 'Berlin'])
        >>> results = model(['paris city'], topk=2)
        >>> print(results)
        {'paris city': [{'entity': 'Paris', 'score': 0.82, 'norm_score': 1.0},
                        {'entity': 'London', 'score': 0.35, 'norm_score': 0.43}]}
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DPR Entity Linking model.

        This initialization extends the base class initialization and sets specific model parameters
        for entity linking tasks. It configures the maximum sequence length to 32768 and sets
        the tokenizer padding side to "right".

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super().__init__(
            *args,
            **kwargs,
        )
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"

    def add_eos(self, input_examples: list[str]) -> list[str]:
        """
        Appends EOS (End of Sequence) token to each input example in the list.

        Args:
            input_examples (list[str]): List of input text strings.

        Returns:
            list[str]: List of input texts with EOS token appended to each example.
        """
        input_examples = [
            input_example + self.model.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples

    def __call__(self, ner_entity_list: list, *args: Any, **kwargs: Any) -> dict:
        """
        Execute entity linking for a list of named entities.

        Args:
            ner_entity_list (list): List of named entities to be linked.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            dict: Entity linking results mapping entities to their linked entries.
        """
        ner_entity_list = self.add_eos(ner_entity_list)
        return super().__call__(ner_entity_list, *args, **kwargs)
