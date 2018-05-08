# coding: utf-8

# # 采用TensorFlow 来搭建CNN进行手写数字预测

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os


# <font color = red>首先对数据进行预处理<br>
#     1、将数据图像标准化<br>
#     2、将数据打包成为mini-batch<br>

# In[2]:




class get_mini_batch(object):
    """
    """

    def __init__(self, data, batch_size, feature_num):
        '''
        data:DataFrame数据类型，为读入的数据，第一列为标签，之后为像素数
        batch_size:每一个batch的样本数
        feature_size:每个样本特征数
        '''
        # 提取出标签，和图片数据
        # x_train ,y_train 均为矩阵

        y_train = data.SalePrice.values.astype(np.float32)
        # 将像素归一化为【0:1】
        x_train = data.drop('SalePrice', axis=1).values.astype(np.float32)


        self.x_train = x_train
        self.y_train = y_train
        self.mini_batch_size = batch_size
        self.feature_num = feature_num
        self.index_in_epoch = 0
        self.current_epoch = 0.0
        self.select_array = np.array([])

    def next_batch(self):
        '''
        return: x_train_data_batch , y_train_data_batch
        '''
        start = self.index_in_epoch
        self.index_in_epoch += self.mini_batch_size
        self.current_epoch += self.mini_batch_size / (len(self.x_train))

        # 将选择数组扩充到与训练样本一样的长度
        if not len(self.select_array) == len(self.x_train):
            self.select_array = np.arange(len(self.x_train))

        # r若是第一次取batch，则打乱顺序取
        if start == 0:
            np.random.shuffle(self.select_array)

        # 若到了数据尾部,则打乱重新开始选择
        if self.index_in_epoch > self.x_train.shape[0]:
            start = 0
            np.random.shuffle(self.select_array)
            self.index_in_epoch = self.mini_batch_size
        end = self.index_in_epoch

        # 至此已经选出mini-batch所以要对image进行标准化
        x_tr = self.x_train[self.select_array[start:end]]
        y_tr = self.y_train[self.select_array[start:end]]

        return x_tr, y_tr


# #  建立TensorFlow CNN模型

# In[3]:

#神经网络结构 输入层=============》隐藏层======》dropout层==========》输出层
def inference(input_batch, batch_size, n_feature, dropout1_ratio,dropout2_ratio,dropout3_ratio):
    '''
    参数说明:
    input_batch：输入数据batch [batch_size , n_feature]
    batch_size : 每一批图像数量
    n_feature:   每一个数据的特征总数

    返回值:
     output tensor with the computed logits, float, [batch_size, predice_Value]
    '''

    #hidden  layer1
    with tf.variable_scope('hidden_layer1') as scope:
       #weights 为隐藏层神经元的权重 shape = [输入层神经元个数,隐藏层神经元个数]
        weights = tf.get_variable('weights',
                                  shape = [n_feature,512],
                                   dtype=tf.float32,
                                   initializer=(tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32)))
        biases = tf.get_variable('biases',
                                  shape =[512],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))
        hidden_layer1 = tf.nn.relu(tf.matmul(input_batch,weights)+biases)

    #dropout layer1
    with tf.variable_scope('dropout_layer1') as scope:
       #weights 为隐藏层神经元的权重 shape = [输入层神经元个数,隐藏层神经元个数]
        dropout_ratio1 = tf.constant(dropout1_ratio,dtype=tf.float32)
        drop_out1 = tf.nn.dropout(hidden_layer1,keep_prob=dropout_ratio1,name='dropout1')

    #hidden layer2
    with tf.variable_scope('hidden_layer2') as scope:
       #weights 为隐藏层神经元的权重 shape = [输入层神经元个数,隐藏层神经元个数]
        weights = tf.get_variable('weights',
                                  shape = [512,1024],
                                   dtype=tf.float32,
                                   initializer=(tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32)))
        biases = tf.get_variable('biases',
                                  shape =[1024],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))
        hidden_layer2 = tf.nn.relu(tf.matmul(drop_out1,weights)+biases)

    #dropout layer2
    with tf.variable_scope('dropout_layer2') as scope:
       #weights 为隐藏层神经元的权重 shape = [输入层神经元个数,隐藏层神经元个数]
        dropout_ratio2 = tf.constant(dropout2_ratio,dtype=tf.float32)
        drop_out2 = tf.nn.dropout(hidden_layer2,keep_prob=dropout_ratio2,name='dropout2')

    #hidden layer3
    with tf.variable_scope('hidden_layer3') as scope:
       #weights 为隐藏层神经元的权重 shape = [输入层神经元个数,隐藏层神经元个数]
        weights = tf.get_variable('weights',
                                  shape = [1024,512],
                                   dtype=tf.float32,
                                   initializer=(tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32)))
        biases = tf.get_variable('biases',
                                  shape =[512],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))
        hidden_layer3 = tf.nn.relu(tf.matmul(drop_out2,weights)+biases)

    #dropout layer3
    with tf.variable_scope('dropout_layer3') as scope:
       #weights 为隐藏层神经元的权重 shape = [输入层神经元个数,隐藏层神经元个数]
        dropout_ratio3 = tf.constant(dropout3_ratio,dtype=tf.float32)
        drop_out3 = tf.nn.dropout(hidden_layer3,keep_prob=dropout_ratio3,name='dropout2')

    #output layer
    with tf.variable_scope('output_layer') as scope:
        # weights 为隐藏层神经元的权重 shape = [输入层神经元个数,隐藏层神经元个数]
        weights = tf.get_variable('weights',
                                  shape=[512, 1],
                                  dtype=tf.float32,
                                  initializer=(tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)))
        biases = tf.get_variable('biases',
                                 shape=[1],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        output = tf.nn.relu(tf.matmul(drop_out3, weights) + biases)

    return output

# In[4]:


def losses(logits, labels):
    '''
    参数说明:
    logits:logits tensor ,float ,[batch_size]
    labels:label  tensor ,tf.int32,[batch_size]

    返回值:
    loss tensor float type

    '''

    with tf.variable_scope('loss') as scope:
       MSE = tf.square(labels-logits)
       loss = tf.reduce_mean(MSE)
    return loss


# In[5]:


def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# In[6]:


# def evaluation(logits, labels):
#     with tf.variable_scope('accuracy') as scope:
#         correct = tf.nn.in_top_k(logits, labels, 1)
#         correct = tf.cast(correct, tf.float16)
#         accuracy = tf.reduce_mean(correct)
#         tf.summary.scalar(scope.name + '/accuracy', accuracy)
#     return accuracy


# #  对模型进行训练

# In[7]:


# 每个minibatch的图片数量
BATCH_SIZE = 1400


MAX_STEP = 500000
learning_rate = 0.0001


# In[8]:


def run_training():
    train_data = pd.read_csv('data/train_data_process.csv')
    logs_train_dir = 'logs/train/'

    N_FEATURES = train_data.shape[1]-1

    with tf.Graph().as_default():

        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_FEATURES])
        y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE])

        batch_generater = get_mini_batch(train_data, BATCH_SIZE,N_FEATURES)
        logits = inference(x, BATCH_SIZE, N_FEATURES, 0.75,0.75,0.75)

        loss = losses(logits, y_)
        # acc = evaluation(logits, y_)
        train_op = trainning(loss, learning_rate=learning_rate)

        with tf.Session()  as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            # summary_op =tf.summary.merge_all()
            # train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
            print('Start Trainning')
            try:
                for step in np.arange(MAX_STEP):
                    X_train, y_train = batch_generater.next_batch()
                    ll,_,tra_loss = sess.run([logits,train_op,loss],feed_dict={x:X_train,y_:y_train})

                    if step % 50 == 0:
                        print('Step %d, train loss = %.2f' % (step, tra_loss))
                        # with tf.Graph().as_default():
                        # summary_str = sess.run(summary_op)
                        # train_writer.add_summary(summary_str, step)

                    if step % 2000 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
            except tf.errors.OutOfRangeError:
                print('Done trainning -- epoch limit reached')


class get_test_data(object):
    def __init__(self, data, batch_size, feature_num):
        '''
        data:DataFrame数据类型，为读入的数据，第一列为标签，之后为像素数
        batch_size:每一个batch的图片数
        '''
        # 提取出标签，和图片数据
        # x_train 为矩阵


        x_train = data.values.astype(np.float32)


        self.x_train = x_train
        self.mini_batch_size = batch_size
        self.feature_num = feature_num
        self.num = x_train.shape[0]
        self.index = 0

    def next_batch(self):
        '''
        return: x_train_data_batch , y_train_data_batch
        '''
        start = self.index
        end = self.index + self.mini_batch_size
        if end > self.num:
            end = self.num
            x_tr = self.x_train[start:end]
            self.index = self.num
            return x_tr
        x_tr = self.x_train[start:end]
        self.index += self.mini_batch_size
        return x_tr

    def data_is_over(self):
        if self.index == self.num:
            return True
        else:
            return False


# In[9]:

def predict_the_test_data():
    test_data = pd.read_csv('data/test_data_process.csv')
    logs_train_dir = 'logs/train/'
    TEST_BATCH_SIZE = 1
    TEST_N_FEATHURE = test_data.shape[1]
    test_data_generator = get_test_data(test_data,TEST_BATCH_SIZE,TEST_N_FEATHURE)

    with tf.Graph().as_default():

        x_test = tf.placeholder(tf.float32, shape=[TEST_BATCH_SIZE,TEST_N_FEATHURE])
        logits = inference(x_test, TEST_BATCH_SIZE, TEST_N_FEATHURE, 1, 1,1)
        saver = tf.train.Saver()
        with tf.Session() as sess:

            print("读取保存节点，导入参数模型....")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)

            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("载入成功,global_step is %s" % global_step)


            else:
                print('载入失败')

            predict_label = np.array([])


            while (test_data_generator.data_is_over() == False):
                X_ = test_data_generator.next_batch()
                prediction = sess.run(logits, feed_dict={x_test: X_})
                predict_price =  np.expm1(prediction)
                predict_label = np.append(predict_label,predict_price)
                #print('完成4000个样本预测')

            print("完成全部预测")
            print(predict_label.shape)
            submission = pd.DataFrame({'id':test_data.index+1461,'SalePrice':predict_label},columns=['id','SalePrice'])
            submission.to_csv('data/cnn_sub.csv', index=False, sep=',')


#run_training()
predict_the_test_data()