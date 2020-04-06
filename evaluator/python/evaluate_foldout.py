"""
@author: Zhongchuan Sun
"""
import itertools
import numpy as np
import sys
import heapq
from concurrent.futures import ThreadPoolExecutor
@profile
def argmax_top_k(a, top_k=50):
    ele_idx = np.argpartition(-a, top_k)[:top_k]
    idx = np.argsort(a[ele_idx])
    ele_idx = ele_idx[np.flip(idx)]
    # ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    # return np.array([idx for ele, idx in ele_idx], dtype=np.intc)
    return ele_idx

@profile
def precision(hits, lenth): # (rank, ground_truth):
    # hits = [1 if item in ground_truth else 0 for item in rank]
    # result = np.cumsum(hits, dtype=np.float)/np.arange(1, len(rank)+1)
    # return result
    result = np.cumsum(hits, dtype=np.float) / np.arange(1, lenth + 1)
    return result

@profile
def recall(hits, lenth): # (rank, ground_truth):
    # hits = [1 if item in ground_truth else 0 for item in rank]
    # result = np.cumsum(hits, dtype=np.float) / len(ground_truth)
    # return result
    result = np.cumsum(hits, dtype=np.float) / lenth
    return result


@profile
def map(hits, pre, len_rank, len_gt): # (rank, ground_truth, pre):
    # pre = precision(rank, ground_truth)
    # pre = [pre[idx] if item in ground_truth else 0 for idx, item in enumerate(rank)]
    pre = [pre[idx] if hits[idx] else 0 for idx in range(len_rank)]
    sum_pre = np.cumsum(pre, dtype=np.float32)
    # gt_len = len(ground_truth)
    # len_rank = np.array([min(i, gt_len) for i in range(1, len(rank)+1)])
    result = sum_pre/len_gt
    return result

@profile
def ndcg(hits, len_rank, len_gt): # (rank, ground_truth):
    # len_rank = len(rank)
    # len_gt = len(ground_truth)
    # idcg_len = min(len_gt, len_rank)
    #
    # # calculate idcg
    # idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    # idcg[idcg_len:] = idcg[idcg_len-1]
    #
    # # idcg = np.cumsum(1.0/np.log2(np.arange(2, len_rank+2)))
    # dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
    # result = dcg/idcg
    # return result
    idcg_len = min(len_gt, len_rank)
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len - 1]
    dcg = np.cumsum([1.0 / np.log2(idx + 2) if hits[idx] else 0.0 for idx in range(len_rank)])
    result = dcg/idcg
    return result


@profile
def mrr(hits, lenth): # (rank, ground_truth):
    # last_idx = sys.maxsize
    # for idx, item in enumerate(rank):
    #     if item in ground_truth:
    #         last_idx = idx
    #         break
    # result = np.zeros(len(rank), dtype=np.float32)
    # result[last_idx:] = 1.0/(last_idx+1)
    # return result
    try:
        last_idx = hits.index(1)
    except ValueError:
        last_idx = sys.maxsize
    result = np.zeros(lenth, dtype=np.float32)
    result[last_idx:] = 1.0 / (last_idx + 1)
    return result


def eval_score_matrix_foldout(score_matrix, test_items, top_k=50, thread_num=None):
    @profile
    def _eval_one_user(idx):
        scores = score_matrix[idx]  # all scores of the test user
        test_item = test_items[idx]  # all test items of the test user
        ranking = argmax_top_k(scores, top_k)  # Top-K items
        hits = [1 if item in test_item else 0 for item in ranking]
        # p = precision(ranking, test_item)
        # r = recall(ranking, test_item)
        # m = map(ranking, test_item, p)
        # n = ndcg(ranking, test_item)
        # mr = mrr(ranking, test_item)
        p = precision(hits, top_k)
        r = recall(hits, len(test_item))
        m = map(hits, p, top_k, len(test_item))
        n = ndcg(hits, top_k, len(test_item))
        mr = mrr(hits, top_k)
        result = [p, r, m, n, mr]

        result = np.array(result, dtype=np.float32).flatten()
        return result

    # with ThreadPoolExecutor(max_workers=thread_num) as executor:
    #     batch_result = executor.map(_eval_one_user, range(len(test_items)))
    result = []
    for i in range(len(test_items)):
        result.append(_eval_one_user(i))

    # result = list(batch_result)  # generator to list
    return np.array(result)  # list to ndarray
