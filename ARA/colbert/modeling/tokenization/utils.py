import torch


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positives, negatives, query_negs, kw_list, bsize):
    assert len(queries) == len(positives) == len(negatives) == len(query_negs) == len(kw_list)
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)
    QN_ids, QN_mask = query_tokenizer.tensorize(query_negs)
    
    for i in range(len(kw_list)):
        kw_list[i][0] = 1
        kw_list[i][-1] = 1
        kw_list[i] = kw_list[i][:Q_ids.shape[1]]
        if Q_ids.shape[1] > len(kw_list[i]):
            kw_list[i].extend([1]*(Q_ids.shape[1] - len(kw_list[i])))
    kw = torch.tensor(kw_list, dtype=torch.float)
    
    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]
    QN_ids, QN_mask = QN_ids[indices], QN_mask[indices]
    

    (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)
    QN_batches = _split_into_batches(QN_ids, QN_mask, bsize)

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask), (qn_ids, qn_mask) in zip(query_batches, positive_batches, negative_batches, QN_batches):
        Q = (torch.cat((q_ids, q_ids, qn_ids)), torch.cat((q_mask, q_mask, qn_mask)))
        D = (torch.cat((p_ids, n_ids, p_ids)), torch.cat((p_mask, n_mask, p_mask)))
        kw_weight = torch.cat((kw, kw, kw))
        batches.append((Q, D, kw_weight))

    return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches
