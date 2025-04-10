import os
import torch

def slow_rerank(args, query, pids, passages, keyword):
    colbert = args.colbert
    inference = args.inference
    
    Q = inference.queryFromText([query])

    keyword[0] = 1
    keyword[-1] = 1
    keyword = keyword[:Q.shape[1]]
    if Q.shape[1] > len(keyword):
        keyword.extend([1]*(Q.shape[1] - len(keyword)))
    keyword = torch.tensor(keyword, dtype=torch.float, device=Q.device)

    D_ = inference.docFromText(passages, bsize=args.bsize)
    scores, kw_scores = colbert.score(Q, D_, keyword)
    scores = scores.cpu()

    scores = scores.sort(descending=True)
    ranked = scores.indices.tolist()

    ranked_scores = scores.values.tolist()
    ranked_pids = [pids[position] for position in ranked]
    ranked_passages = [passages[position] for position in ranked]

    assert len(ranked_pids) == len(set(ranked_pids))

    return list(zip(ranked_scores, ranked_pids, ranked_passages))
