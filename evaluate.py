'''
Top-K recommendation 性能评估 :
方法:  leave-1-out evaluation
指标:  Hit Ratio -- 命中率  NDCG -- 归一化折损累计增益
'''
import math
import heapq  # for retrieval topK
import multiprocessing # 多线程
import numpy as np
from time import time

# 多进程环境中的每个进程都有自己的内存空间，因此全局变量不会在进程之间共享。你需要在每个进程中重新初始化这些变量。
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
  '''
  Arguments:
    model: 模型
    testRatings: 测试评分
    testNegatives: 测试负样本
    K: Top-K
    num_thread: 线程数
  '''
  global _model
  global _testRatings
  global _testNegatives
  global _K
  _model = model
  _testRatings = testRatings
  _testNegatives = testNegatives
  _K = K

  hits, ndcgs = [], [] # 命中率，归一化折损累计增益
  # 多线程
  if (num_thread > 1):
    pool = multiprocessing.Pool(processes=num_thread)
    res = pool.map(eval_one_rating, range(len(_testRatings)))
    pool.close()
    pool.join()
    hits = [r[0] for r in res]
    ndcgs = [r[1] for r in res]
    return (hits, ndcgs)
  
  # 单线程
  for idx in range(len(_testRatings)):
    (hr, ndcg) = eval_one_rating(idx)
    hits.append(hr)
    ndcgs.append(ndcg)
  return (hits, ndcgs)

def eval_one_rating(idx):
  '''
  idx: 索引
  '''
  rating = _testRatings[idx]
  items = _testNegatives[idx]
  u = rating[0]
  gtItem = rating[1]
  items.append(gtItem)

  # Get prediction scores
  map_item_score = {}
  users = np.full(len(items), u, dtype='int32')
  predictions = _model.predict([users, np.array(items)], batch_size=100, verbose=0)
  for i in range(len(items)):
    item = items[i]
    map_item_score[item] = predictions[i]
  items.pop()

  # Evaluate top rank list
  ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get) # 根据字典的值选取前K个
  hr = getHitRatio(ranklist, gtItem)
  ndcg = getNDCG(ranklist, gtItem)
  return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
  for item in ranklist:
    if item == gtItem:
      return 1
  return 0

def getNDCG(ranklist, gtItem):
  for i in range(len(ranklist)):
    item = ranklist[i]
    if item == gtItem:
      return math.log(2) / math.log(i + 2) # ndcg
  return 0

