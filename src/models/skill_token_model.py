"""
Implementation of skill neologisms for Hugging Face models.

A skill neologism is a set of soft tokens integrated into the model's vocabulary
and optimized to improve model capabilities on a specific skill, without modifying
model weights. This module provides:

- ``SkillTokenModel``: main class for registering skills, managing their embeddings,
  and handling tokenizer/model updates.
- ``insert_skill_tokens``: helper to inject skill token placeholders into text prompts.
- ``load_skill_model_from_cfg``: utility to instantiate a ``SkillTokenModel`` from a
  config dict, optionally loading a saved skill checkpoint.

Skill embeddings are saved and loaded independently of the full model vocabulary
(``embeddings.pt`` + ``metadata.yaml``), making checkpoints lightweight.
"""

import os
import re
from typing import Dict, List, Optional

import torch
import yaml

from sequence_map_experiment.config import OUTPUT_DIR
from sequence_map_experiment.model import load_any_model
from src.models.loading import get_embedding_weights, load_base_hf_model
from src.models.model_utils import get_mean_emb


class SkillTokenModel:
    """Manage skill tokens (groups of special tokens) and their embeddings.

    Each skill is represented as a sequence of soft tokens added to the model's
    vocabulary and embedding matrix. Model weights are kept frozen; only the skill
    token embeddings are trained.

    Assumes a causal LM (``AutoModelForCausalLM``) but generalises to other LM
    types with minor changes.

    Args:
        model_name: Identifier for the base model.
        device: Target device. Defaults to CUDA if available, otherwise CPU.
        model: Optional pre-loaded model. Must be provided together with ``tokenizer``.
        tokenizer: Optional pre-loaded tokenizer. Must be provided together with ``model``.
        embedding_weights_str: Optional string expression evaluated to resolve the
            embedding weight tensor.
    """

    def __init__(
        self,
        model_name: str,
        device: str = None,
        model=None,
        tokenizer=None,
        embedding_weights_str: str = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model is None or tokenizer is None:
            self.model, self.tokenizer, self.device = load_base_hf_model(
                model_name,
                device=self.device,
                eval_mode=False,
                resize_embeds_for_pad=False,
            )
        else:
            if model is None or tokenizer is None:
                raise ValueError("Provide both model and tokenizer, or neither.")
            print(f"Using provided tokenizer and model for {model_name} on {self.device}...")
            self.tokenizer = tokenizer
            self.model = model
            self.model.to(self.device, dtype=torch.bfloat16)

        self.model.to(self.device)

        if embedding_weights_str is not None:
            self.embedding_weights = eval(embedding_weights_str)
        else:
            self.embedding_weights = get_embedding_weights(self.model)
        self.embedding_weights.requires_grad = True

        # Registry: skill_id -> {desc, length, tokens, token_ids}
        self.skill_tokens: Dict[str, Dict] = {}
        self.model_cfg = {}

    # Internal helpers
    def _add_tokens_to_tokenizer(self, token_strings: List[str]) -> List[int]:
        """Add tokens to the tokenizer (skipping any already present) and return their ids."""
        vocab = self.tokenizer.get_vocab()
        to_add = [t for t in token_strings if t not in vocab]
        if to_add:
            print(f"Adding {len(to_add)} new tokens to tokenizer: {to_add}")
            self.tokenizer.add_tokens(to_add)
            self.model.resize_token_embeddings(len(self.tokenizer))
        return [self.tokenizer.convert_tokens_to_ids(t) for t in token_strings]

    def _get_embedding_weight(self) -> torch.nn.Parameter:
        """Return the model's input embedding weight matrix."""
        return self.model.get_input_embeddings().weight

    def _expand_skill_tokens(self, text: str) -> str:
        """Expand ``<|skill_id|>`` placeholders into their per-token form.

        Example: ``<|SHIFT|>`` with length 3 becomes
        ``<|SHIFT-0|><|SHIFT-1|><|SHIFT-2|>``.
        """
        for skill_name, meta in self.skill_tokens.items():
            length = meta["length"]
            skill_name_escape = skill_name.replace("[", "\\[").replace("]", "\\]")
            pattern = f"<\\|{skill_name_escape}\\|>"
            replacement = "".join(f"<|{skill_name}-{i}|>" for i in range(length))
            text = re.sub(pattern, replacement, text)
        return text

    # Public API
    def expand_dataset(self, dataset: List[str]) -> List[str]:
        """Expand skill token placeholders in every text entry of a dataset.

        Args:
            dataset: List of raw text strings.

        Returns:
            List of text strings with skill placeholders fully expanded.
        """
        for skill_token in self.skill_tokens:
            print(f"Expanding skill token {skill_token} in dataset texts...")
            dataset = [t.replace(skill_token, f"<|{skill_token}|>") for t in dataset]
            dataset = [self._expand_skill_tokens(t) for t in dataset]
        return dataset

    def get_skill_token_embeddings(self, skill_name: str) -> torch.Tensor:
        """Return the embeddings for a skill as a tensor of shape ``(length, hidden_size)``.

        Args:
            skill_name: Registered skill identifier.
        """
        if skill_name not in self.skill_tokens:
            raise ValueError(f"Skill '{skill_name}' not found in registry")
        ids = self.skill_tokens[skill_name]["token_ids"]
        emb = self._get_embedding_weight()
        with torch.no_grad():
            return emb[ids].detach().cpu()

    def set_skill_token_embeddings(self, skill_name: str, embeddings: torch.Tensor):
        """Set the embeddings for a skill from a tensor of shape ``(length, hidden_size)``.

        Args:
            skill_name: Registered skill identifier.
            embeddings: New embedding values.
        """
        if skill_name not in self.skill_tokens:
            raise ValueError(f"Skill '{skill_name}' not found in registry")
        ids = self.skill_tokens[skill_name]["token_ids"]
        emb = self._get_embedding_weight()
        if embeddings.shape[0] != len(ids):
            raise ValueError(f"Embeddings length {embeddings.shape[0]} does not match skill length {len(ids)}")
        with torch.no_grad():
            emb[ids] = embeddings.to(emb.device)

    def create_skill(
        self,
        skill_id: str,
        length: int,
        desc: str = "",
        init_method: str = "rand",
        init_token: str = None,
    ) -> Dict:
        """Register a new skill and add its token rows to the tokenizer and embedding matrix.

        Args:
            skill_id: Unique identifier for the skill (e.g. ``"SHIFT"``).
            length: Number of soft tokens representing this skill.
            desc: Human-readable description (stored in metadata).
            init_method: One of ``"rand"`` (random normal), ``"from_token"`` (copy from an
                existing token), or ``"from_pretrain_skills"`` (mean of pretraining op embeddings).
            init_token: Source token string, required when ``init_method="from_token"``.

        Returns:
            Metadata dict for the created skill.
        """
        if skill_id in self.skill_tokens:
            raise ValueError(f"Skill '{skill_id}' already exists")

        token_strs = _skill_token_names(skill_id, length)
        token_ids = self._add_tokens_to_tokenizer(token_strs)

        metadata = {
            "desc": desc,
            "length": length,
            "tokens": token_strs,
            "token_ids": token_ids,
        }
        self.skill_tokens[skill_id] = metadata
        print(f"Registered skill '{skill_id}': tokens={token_strs}, ids={token_ids}")

        emb = self._get_embedding_weight()

        if init_method == "rand":
            initializer_range = (
                self.model.config.text_config.initializer_range if hasattr(self.model.config, "text_config") else self.model.config.initializer_range
            )
            with torch.no_grad():
                for tid in token_ids:
                    emb[tid].normal_(mean=0.0, std=initializer_range)
            print(f"Initialised skill embeddings from normal(0, {initializer_range})")

        elif init_method == "from_token":
            if init_token is None:
                raise ValueError("init_token must be provided when init_method='from_token'")
            init_id = self.tokenizer.convert_tokens_to_ids(init_token)
            with torch.no_grad():
                init_emb = emb[init_id]
                for tid in token_ids:
                    emb[tid] = init_emb.clone()
            print(f"Initialised skill embeddings from token '{init_token}' (id={init_id})")

        elif init_method == "from_pretrain_skills":
            from sequence_map_experiment.data import PRETRAIN_OPS

            mean_emb = get_mean_emb(emb, self.tokenizer, word_list=PRETRAIN_OPS)
            with torch.no_grad():
                for tid in token_ids:
                    emb[tid] = mean_emb.clone()
            print(f"Initialised skill embeddings from mean of {len(PRETRAIN_OPS)} " f"pretrain op embeddings: {PRETRAIN_OPS}")

        else:
            raise ValueError(f"Unknown init_method: '{init_method}'")

        return metadata

    def get_skill_token_ids(self) -> List[int]:
        """Return a flat list of all token IDs corresponding to registered skill tokens."""
        all_ids = []
        for info in self.skill_tokens.values():
            all_ids.extend(self.tokenizer.convert_tokens_to_ids(info["tokens"]))
        return all_ids

    def get_optimizer(self, lr: float = 1e-3) -> torch.optim.Adam:
        """Return an Adam optimizer that updates only the skill token embeddings.

        Args:
            lr: Learning rate.
        """
        return torch.optim.Adam([self.embedding_weights], lr=lr)

    def zero_out_non_skill_grads(self, batch_input_ids: torch.Tensor):
        """Zero gradients for all tokens except skill tokens present in the current batch.

        This ensures that only the skill token embeddings seen in the batch are updated,
        leaving all other embedding rows untouched.

        Args:
            batch_input_ids: ``(batch_size, seq_len)`` tensor of token IDs for the current batch.
        """
        grad = self.embedding_weights.grad
        if grad is None:
            return
        if torch.isnan(grad).any():
            print("Warning: gradient contains NaN values; skipping zeroing step.")
            return

        device = grad.device
        all_skill_ids = set(self.get_skill_token_ids())
        present_ids = set(torch.unique(batch_input_ids.flatten()).tolist())
        active_ids = list(all_skill_ids & present_ids)

        mask = torch.zeros(grad.shape[0], dtype=torch.bool, device=device)
        mask[active_ids] = True
        grad[~mask] = 0

    def set_skill_unembed_to_zero(self, verbose: bool = False):
        """Zero out unembedding rows for all skill tokens to prevent generating them at inference.

        Args:
            verbose: If True, log each token being zeroed.
        """
        self.model.lm_head.weight = torch.nn.Parameter(self.model.lm_head.weight.data.clone())
        for st, st_dict in self.skill_tokens.items():
            for st_id in st_dict["token_ids"]:
                if verbose:
                    print(f"Zeroing unembedding for skill token '{st}' (id={st_id})")
                self.model.lm_head.weight.data[st_id, :] = 0.0

    def get_skill_tokens_embeddings(self) -> Dict:
        """Return skill embeddings as a dict of numpy arrays keyed by skill id."""
        emb = self._get_embedding_weight()
        skill_embeddings = {}
        for sid, info in self.skill_tokens.items():
            with torch.no_grad():
                skill_embeddings[sid] = emb[info["token_ids"]].detach().cpu().numpy()
        return skill_embeddings

    # Saving and loading
    def save_skills(self, checkpoint_dir: str):
        """Save all registered skills' embeddings and YAML metadata to disk.

        Directory layout::

            checkpoint_dir/
                embeddings.pt    # dict: skill_id -> tensor (length × hidden)
                metadata.yaml    # dict: skill_id -> {desc, length, tokens}

        Args:
            checkpoint_dir: Path to the output directory (created if absent).
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        emb = self._get_embedding_weight()
        save_dict = {}
        meta = {}
        for sid, info in self.skill_tokens.items():
            with torch.no_grad():
                save_dict[sid] = emb[info["token_ids"]].detach().cpu()
            meta[sid] = {
                "desc": info["desc"],
                "length": info["length"],
                "tokens": info["tokens"],
            }
        torch.save(save_dict, os.path.join(checkpoint_dir, "embeddings.pt"))
        with open(os.path.join(checkpoint_dir, "metadata.yaml"), "w") as f:
            yaml.safe_dump(meta, f)
        print(f"Saved {len(save_dict)} skill(s) to {checkpoint_dir}")

    def load_skills(self, checkpoint_dir: str, overwrite_existing: bool = False):
        """Load skill embeddings and metadata, inserting them into the tokenizer and model.

        If ``checkpoint_dir`` is not a local directory, attempts to download from the
        Hugging Face Hub using it as a repo id.

        Tokens not yet present in the tokenizer are added automatically. If a skill id
        already exists in the registry and ``overwrite_existing=False``, it is skipped.

        Args:
            checkpoint_dir: Local directory path or Hugging Face Hub repo id.
            overwrite_existing: If True, overwrite skills already present in the registry.
        """
        if not os.path.exists(checkpoint_dir):
            from huggingface_hub import hf_hub_download

            print(f"'{checkpoint_dir}' not found locally; downloading from Hugging Face Hub...")
            meta_path = hf_hub_download(repo_id=checkpoint_dir, filename="metadata.yaml")
            emb_path = hf_hub_download(repo_id=checkpoint_dir, filename="embeddings.pt")
            checkpoint_dir = os.path.dirname(meta_path)
            print(f"Downloaded to {checkpoint_dir}")
        else:
            meta_path = os.path.join(checkpoint_dir, "metadata.yaml")
            emb_path = os.path.join(checkpoint_dir, "embeddings.pt")

        if not os.path.exists(meta_path) or not os.path.exists(emb_path):
            raise FileNotFoundError(f"Expected 'metadata.yaml' and 'embeddings.pt' in '{checkpoint_dir}'")

        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
        saved = torch.load(emb_path, map_location="cpu")

        # Add any missing tokens to the tokenizer before writing embeddings
        all_token_strs = [t for info in meta.values() for t in info["tokens"]]
        self._add_tokens_to_tokenizer(all_token_strs)

        emb = self._get_embedding_weight()
        for sid, info in meta.items():
            if sid in self.skill_tokens and not overwrite_existing:
                print(f"Skipping existing skill '{sid}' (set overwrite_existing=True to force).")
                continue
            token_strs = info["tokens"]
            token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in token_strs]
            self.skill_tokens[sid] = {
                "desc": info.get("desc", ""),
                "length": info.get("length", len(token_ids)),
                "tokens": token_strs,
                "token_ids": token_ids,
            }
            with torch.no_grad():
                emb[token_ids] = saved[sid].to(self.device)

        print(f"Loaded skills from '{checkpoint_dir}'. Registry now has {len(self.skill_tokens)} skill(s).")


# Utilities
def _skill_token_names(skill_id: str, length: int) -> List[str]:
    """Return the list of textual token names for a skill.

    Example: ``skill_id='SHIFT', length=3`` → ``['<|SHIFT-0|>', '<|SHIFT-1|>', '<|SHIFT-2|>']``.
    A length of 1 still produces ``'<|SHIFT-0|>'`` for consistent indexing.

    Args:
        skill_id: Skill identifier string.
        length: Number of tokens (must be >= 1).
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    return [f"<|{skill_id}-{i}|>" for i in range(length)]


def load_skill_model_from_cfg(cfg, skill_checkpoint_path: str = None) -> SkillTokenModel:
    """Instantiate a ``SkillTokenModel`` from a config dict.

    Loads the base model from ``cfg.pretrained.checkpoint`` (local or Hub), then either
    loads skills from ``skill_checkpoint_path`` or creates a fresh skill defined in
    ``cfg.skill``.

    Args:
        cfg: Config object with at least ``cfg.pretrained.checkpoint`` and ``cfg.skill``
            fields.
        skill_checkpoint_path: Path to a saved skill checkpoint directory. If None,
            falls back to ``cfg.skill.skill_checkpoint_path`` if present, then creates
            a new skill from cfg.

    Returns:
        A fully initialised ``SkillTokenModel``.
    """
    if skill_checkpoint_path is None:
        skill_checkpoint_path = cfg.get("skill", {}).get("skill_checkpoint_path", None)

    pretrained_path = cfg.pretrained.checkpoint
    local_model_path = os.path.join(OUTPUT_DIR, pretrained_path)
    if os.path.exists(local_model_path):
        model_dir = local_model_path
        is_local = True
    else:
        model_dir = pretrained_path
        is_local = False

    saved_cfg = {}
    if is_local:
        config_path = os.path.join(model_dir, "..", "..", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved_cfg = yaml.safe_load(f)

    model, tokenizer, model_name = load_any_model(model_dir, saved_cfg)

    adapter = SkillTokenModel(model_name=model_name, model=model, tokenizer=tokenizer)
    adapter.model_cfg = saved_cfg

    if skill_checkpoint_path is not None:
        if not os.path.exists(skill_checkpoint_path):
            raise FileNotFoundError(f"Skill checkpoint not found: '{skill_checkpoint_path}'")
        adapter.load_skills(skill_checkpoint_path, overwrite_existing=True)
    else:
        adapter.create_skill(
            skill_id=cfg.skill.name,
            length=cfg.skill.length,
            desc=cfg.skill.description,
            init_method=cfg.skill.get("init_method", "rand"),
            init_token=cfg.skill.get("init_token", None),
        )

    return adapter


def insert_skill_tokens(
    texts: List[str],
    skill_name: str,
    replace_str: str,
    adapter: SkillTokenModel,
    ignore_str: Optional[List[str]] = None,
) -> List[str]:
    """Replace occurrences of ``replace_str`` in texts with expanded skill tokens.

    Substrings listed in ``ignore_str`` are temporarily protected from replacement.

    Args:
        texts: Input text strings.
        skill_name: Registered skill identifier used to look up token length.
        replace_str: Literal string in the texts to be replaced by the skill tokens.
        adapter: ``SkillTokenModel`` instance holding the skill registry.
        ignore_str: Optional list of substrings to leave untouched during replacement.

    Returns:
        List of text strings with skill tokens inserted and expanded.
    """
    ignore_str = ignore_str or []
    out = []
    nb_replacements = 0

    for t in texts:
        # Temporarily protect ignored substrings
        protected = {}
        for i, s in enumerate(ignore_str):
            placeholder = f"<<IGNORE_{i}>>"
            if s in t:
                t = t.replace(s, placeholder)
                protected[placeholder] = s

        nb_replacements += t.count(replace_str)
        t = t.replace(replace_str, f"<|{skill_name}|>")
        t = adapter._expand_skill_tokens(t)

        # Restore protected substrings
        for placeholder, s in protected.items():
            t = t.replace(placeholder, s)

        out.append(t)

    print(f"Inserted skill tokens for '{skill_name}' by replacing '{replace_str}'. " f"Total replacements: {nb_replacements}")
    return out
