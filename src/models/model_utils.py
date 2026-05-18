import torch


def get_mean_emb(emb, tokenizer, word_list):
    """
    Compute mean embedding for a list of words.
    If a word token is not in the tokenizer vocabulary, tokenize it and average
    the embeddings of its sub-tokens.

    Args:
        emb: Embedding matrix (indexable by token id).
        tokenizer: Tokenizer with `tokenize` and `convert_tokens_to_ids`.
        word_list: Iterable of word strings to compute mean embedding for.

    Returns:
        torch.Tensor: Mean embedding vector.
    """
    init_embs = []
    for op in word_list:
        op_id = tokenizer.convert_tokens_to_ids(op)
        if op_id is None:
            print(f"Pretrain op token '{op}' not found in vocab; tokenizing and averaging embeddings of sub-tokens.")
            sub_tokens = tokenizer.tokenize(op)
            sub_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in sub_tokens]
            sub_embs = []
            for stid in sub_token_ids:
                with torch.no_grad():
                    sub_embs.append(emb[stid].clone())
            mean_sub_emb = torch.stack(sub_embs, dim=0).mean(dim=0)
            init_embs.append(mean_sub_emb)
        else:
            with torch.no_grad():
                init_embs.append(emb[op_id].clone())
    mean_emb = torch.stack(init_embs, dim=0).mean(dim=0)
    return mean_emb
