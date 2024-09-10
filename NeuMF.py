import numpy as np

# import theano
# import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l2
from keras.models import Model
from keras.layers.core import Dense
from keras.layers import Embedding, Input, Dense, Flatten, Multiply
# from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from Dataset import Dataset
from evaluate import evaluate_model
from time import time
import sys
import argparse
import GMF, MLP

#################### Arguments ####################

def parse_args():
  parser = argparse.ArgumentParser(description="Run NeuMF.")
  parser.add_argument('--path', nargs='?', default='自己实现/Data/', help='Input data path.')
  parser.add_argument('--dataset', nargs='?', default='ml-1m', help='Choose a dataset.')
  parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
  parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
  parser.add_argument('--num_factors', type=int, default=8, help='Embedding size.')
  parser.add_argument('--layers', nargs='?', default='[64,32,16,8]', help='Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.')
  parser.add_argument('--reg_mf', type=float, default=0, help="Regularization for MF embeddings.")
  parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]', help="Regularization for each layer")
  parser.add_argument('--num_neg', type=int, default=4, help='Number of negative instances to pair with a positive instance.')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
  parser.add_argument('--learner', nargs='?', default='adam', help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
  parser.add_argument('--verbose', type=int, default=1, help='Show performance per X iterations')
  parser.add_argument('--out', type=int, default=1, help='Whether to save the trained model.')
  parser.add_argument('--mf_pretrain', nargs='?', default='', help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
  parser.add_argument('--mlp_pretrain', nargs='?', default='', help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
  return parser.parse_args()

def init_normal(shape, dtype=None):
  initializer = initializers.initializers_v2.RandomNormal(mean=0.0, stddev=0.01) # 修改为最新语法
  return initializer(shape)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
  assert len(layers) == len(reg_layers)
  num_layer = len(layers) # Number of layers in the MLP
  # Input variables
  user_input = Input(shape=(1, ), dtype='int32', name='user_input')
  item_input = Input(shape=(1, ), dtype='int32', name='item_input')

  MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user', embeddings_initializer=init_normal, embeddings_regularizer=l2(reg_mf), input_length=1)
  MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item', embeddings_initializer=init_normal, embeddings_regularizer=l2(reg_mf), input_length=1)

  MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0]//2, name='mlp_embedding_user', embeddings_initializer=init_normal, embeddings_regularizer=l2(reg_layers[0]), input_length=1)
  MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0]//2, name='mlp_embedding_item', embeddings_initializer=init_normal, embeddings_regularizer=l2(reg_layers[0]), input_length=1) # 使用整除

  # MF part
  mf_user_latent = Flatten()(MF_Embedding_User(user_input))
  mf_item_latent = Flatten()(MF_Embedding_Item(item_input)) 
  mf_vector = Multiply()([mf_user_latent, mf_item_latent])

  # MLP part
  mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
  mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input)) 
  mlp_vector = Multiply()([mlp_user_latent, mlp_item_latent])
  for idx in range(1, num_layer):
    layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
    mlp_vector = layer(mlp_vector)

  # Concatenate MF and MLP parts
  predict_vector = Multiply()([mf_vector, mlp_vector]) # 两个嵌入层的输出进行逐元素相乘

  # Final prediction layer
  prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)

  model = Model(inputs=[user_input, item_input], outputs=prediction) # 定义模型的输入和输出

  return model 

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
  # MF embeddings
  gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
  gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
  model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
  model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

  # MLP embeddings
  mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
  mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
  model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
  model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

  # MLP layers
  for i in range(1, num_layers):
    mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
    model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

  # Prediction weights
  gmf_prediction = gmf_model.get_layer('prediction').get_weights()
  mlp_prediction = mlp_model.get_layer('prediction').get_weights()
  new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
  new_b = gmf_prediction[1] + mlp_prediction[1]
  model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])

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
  mf_dim = args.num_factors
  layers = eval(args.layers)
  reg_mf = args.reg_mf
  reg_layers = eval(args.reg_layers)
  num_negatives = args.num_neg
  learner = args.learner
  learning_rate = args.lr
  num_epochs = args.epochs
  batch_size = args.batch_size
  verbose = args.verbose
  mf_pretrain = args.mf_pretrain
  mlp_pretrain = args.mlp_pretrain

  topK = 10
  evaluation_threads = 1 # mp.cpu_count()
  print("NeuMF arguments: %s" % (args))
  model_out_file = '自己实现/Pretrain/%s_NeuMF_%d_%s_%d.h5' % (args.dataset, mf_dim, args.layers, time())

  # Loading data
  t1 = time()
  dataset = Dataset(args.path + args.dataset)
  train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
  num_users, num_items = train.shape
  print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" % (time()-t1, num_users, num_items, train.nnz, len(testRatings)))

  # Build model
  model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
  if learner.lower() == "adagrad":
    model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='binary_crossentropy')
  elif learner.lower() == "rmsprop":
    model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='binary_crossentropy')
  elif learner.lower() == "adam":
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
  else:
    model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy')

  # Load pretrain model
  if mf_pretrain != '' and mlp_pretrain != '':
    gmf_model = GMF.get_model(num_users, num_items, mf_dim)
    gmf_model.load_weights(mf_pretrain)
    mlp_model = MLP.get_model(num_users, num_items, layers, reg_layers)
    mlp_model.load_weights(mlp_pretrain)
    model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
    print("Load pretrained GMF (%s) and MLP (%s) models done. " % (mf_pretrain, mlp_pretrain))

  # Check Init performance
  (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
  hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
  print("Init: HR = %.4f, NDCG = %.4f" % (hr, ndcg))
  best_hr, best_ndcg, best_iter = hr, ndcg, -1
  if args.out > 0:
    model.save_weights(model_out_file, overwrite=True)

  # Train model
  for epoch in range(num_epochs):
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
    print("The best NeuMF model is saved to %s" % model_out_file)
