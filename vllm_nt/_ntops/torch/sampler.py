import torch


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    if k is None and p is None:
        return logits

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        k = k.to(device=logits.device, dtype=torch.long)
        k = torch.clamp(k, min=1, max=logits.shape[-1])
        top_k_idx = logits_sort.shape[-1] - k
        top_k_threshold = logits_sort.gather(1, top_k_idx.unsqueeze(1))
        logits_sort.masked_fill_(logits_sort < top_k_threshold, -float("inf"))

    if p is not None:
        p = p.to(device=logits.device, dtype=torch.float32)
        probs_sort = logits_sort.softmax(dim=-1, dtype=torch.float32)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    q = torch.empty_like(probs)
    q.exponential_()
    for i, generator in generators.items():
        q[i].exponential_(generator=generator)
    return torch.div(probs, q).argmax(dim=-1).view(-1)
