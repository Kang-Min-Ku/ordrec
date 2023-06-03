import math

def unlist_dic_value(dic):
    new_val = [v[0] for v in dic.values()]
    new_dic = dict(zip(dic.keys(), new_val))
    
    return new_dic

def compute_metrics(pred_u, target_u, top_k):
    pred_k = pred_u[:top_k]
    num_target_items = len(target_u)
    # print(pred_k)
    # print(target_u)
    hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_u]
    num_hits = len(hits_k)

    idcg_k = 0.0
    for i in range(1, min(num_target_items, top_k) + 1):
        idcg_k += 1 / math.log(i + 1, 2)

    dcg_k = 0.0
    for idx, item in hits_k:
        dcg_k += 1 / math.log(idx + 1, 2)

    prec_k = num_hits / top_k
    recall_k = num_hits / min(num_target_items, top_k)
    ndcg_k = dcg_k / idcg_k

    return prec_k, recall_k, ndcg_k