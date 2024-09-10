'''
Generalized Matrix Factorization
'''
import numpy as np
# import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.models import Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Reshape, Flatten, Multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from Dataset import Dataset
from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse


#################### Arguments ####################
def parse_args():
  parser = argparse.ArgumentParser(description="Run GMF.")
  parser.add_argument('--path', nargs='?', default='自己实现/Data/', help='Input data path.')
  parser.add_argument('--dataset', nargs='?', default='ml-1m', help='Choose a dataset.')
  parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
  parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
  parser.add_argument('--num_factors', type=int, default=8, help='Embedding size.')
  parser.add_argument('--regs', nargs='?', default='[0,0]', help="Regularization for user and item embeddings.")
  parser.add_argument('--num_neg', type=int, default=4, help='Number of negative instances to pair with a positive instance.')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
  parser.add_argument('--learner', nargs='?', default='adam', help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
  parser.add_argument('--verbose', type=int, default=1, help='Show performance per X iterations')
  parser.add_argument('--out', type=int, default=1, help='Whether to save the trained model.')
  return parser.parse_args()

def init_normal(shape, dtype=None):
  initializer = initializers.initializers_v2.RandomNormal(mean=0.0, stddev=0.01) # 修改为最新语法
  return initializer(shape)

def get_model(num_users, num_items, latent_dim, regs=[0, 0]):
  # Input variables
  user_input = Input(shape=(1, ), dtype='int32', name='user_input')
  item_input = Input(shape=(1, ), dtype='int32', name='item_input')

  MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding', embeddings_initializer=init_normal, embeddings_regularizer=l2(regs[0]), input_length=1)
  MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding', embeddings_initializer=init_normal, embeddings_regularizer=l2(regs[1]), input_length=1)

  # Crucial to flatten an embedding vector!
  user_latent = Flatten()(MF_Embedding_User(user_input))
  item_latent = Flatten()(MF_Embedding_Item(item_input)) # 将嵌入层的输出进行展平处理

  # Element-wise product of user and item embeddings
  predict_vector = Multiply()([user_latent, item_latent]) # 两个嵌入层的输出进行逐元素相乘

  # Final prediction layer
  prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)

  model = Model(inputs=[user_input, item_input], outputs=prediction) # 定义模型的输入和输出

  return model

def get_train_instances(train, num_negatives):
  user_input, item_input, labels = [], [], []
  num_items = train.shape[1] # 原本求用户数，应该是求物品数
  for (u, i) in train.keys():
    # positive instance
    user_input.append(u)
    item_input.append(i)
    labels.append(1)
    # negative instances
    for t in range(num_negatives):
      j = np.random.randint(num_items)
      while (u, j) in train:
        j = np.random.randint(num_items)
      user_input.append(u)
      item_input.append(j)
      labels.append(0)
  return user_input, item_input, labels

if __name__ == '__main__':
  args = parse_args()
  num_factors = args.num_factors
  regs = eval(args.regs)
  num_negatives = args.num_neg
  learner = args.learner
  learning_rate = args.lr
  epochs = args.epochs
  batch_size = args.batch_size
  verbose = args.verbose

  topK = 10
  evaluation_threads = 1 # mp.cpu_count()
  print("GMF arguments: %s" % (args))
  model_out_file = '自己实现/Pretrain/%s_GMF_%d_%d.h5' % (args.dataset, num_factors, time())

  # Loading data
  t1 = time()
  dataset = Dataset(args.path + args.dataset)
  train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
  num_users, num_items = train.shape
  print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" % (time()-t1, num_users, num_items, train.nnz, len(testRatings)))

  # Build model
  model = get_model(num_users, num_items, num_factors, regs)
  if learner.lower() == "adagrad":
    model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='binary_crossentropy')
  elif learner.lower() == "rmsprop":
    model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='binary_crossentropy')
  elif learner.lower() == "adam":
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
  else:
    model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy')

  # Check Init performance
  t2 = time()
  (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
  hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
  print("Init: HR = %.4f, NDCG = %.4f\t [%.1f s]" % (hr, ndcg, time()-t2))

  # Train model
  best_hr, best_ndcg, best_iter = hr, ndcg, -1
  for epoch in range(epochs):
    t3 = time()
    # Generate training instances
    user_input, item_input, labels = get_train_instances(train, num_negatives)

    # Training
    hist = model.fit([np.array(user_input), np.array(item_input)], np.array(labels), batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
    t4 = time()

    # Evaluation
    if epoch % verbose == 0:
      (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
      hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
      print("Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]" % (epoch, t4-t3, hr, ndcg, loss, time()-t4))
      if hr > best_hr:
        best_hr, best_ndcg, best_iter = hr, ndcg, epoch
        if args.out > 0:
          model.save_weights(model_out_file, overwrite=True)

  print("End. Best Iteration %d: HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
  if args.out > 0:
    print("The best GMF model is saved to %s" % model_out_file)







