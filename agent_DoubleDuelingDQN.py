import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os

from env_DoubleDuelingDQN import GameEnv

env = GameEnv(partial=False, size=5)

class Qnetwork():
    def __init__(self, h_size):
        # flattening
        # conv net
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d(inputs = self.imageIn, num_outputs = 32, kernel_size = [8, 8], stride = [4, 4], \
                                 padding= 'VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], \
                                 padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], \
                                 padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], \
                                 padding='VALID', biases_initializer=None)

        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        self.AW = tf.Variable(tf.random_normal([h_size//2, env.actions]))
        self.VW = tf.Variable(tf.random_normal([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        self.Qout = self.Value + tf.subtract(self.Advantage, \
                                             tf.reduce_mean(self.Advantage, axis =1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        self.updatemodel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def processState(states):
    return np.reshape(states, [21168])

def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

batch_size = 32        # 경험 배치의 수 (각 학습 단계에서)
update_freq = 4        # 갱신 주기
discount_factor = 0.99 # 감쇠상수
e_start = 1             # 랜덤한 액션을 취할 가능성
e_end = 0.1             # 랜덤한 액션을 종료할 가능성

annealing_step = 10000  # 랜덤한 액션을 시작할 가능성에서 종료할 가능성으로 도달하는데 설정한 학습 단계 수

total_episodes = 10000  # 전체 에피소드의 수
max_epLength = 50       # 각 episode에서 허용하는 최대 에피소드 길이

pre_train_steps = 10000 # 학습 시작 전 랜덤 액션의 단계 수
load_model = False
path = "./double_dueling_DQN"
h_size = 512            # Advantage/가치 값 각각으로 분리되기 전 마지막 conv 계층의 크기
tau = 0.001             # 제 1 네트워크를 얼마나 활용해 target 네트워크를 업데이트 할지

tf.reset_default_graph()
main_Q_Network = Qnetwork(h_size)
target_Q_Network = Qnetwork(h_size)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# 랜덤 액션이 감소하는 비율 설정
e = e_start
stepDrop = (e_start-e_end)/annealing_step

# 보상의 총 합과 에피소드별 단계 수를 담을 리스트 생성
stepList = []
RList = []
total_steps = 0

# 모델을 저장할 경로를 설정
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model==True:
        print('Loading model..')
        checkpoint = tf.train.get_checkpoint_state(path)
        saver.restore(sess,checkpoint.model_checkpoint_path)
    # 타겟 네트워크가 제1 네트워크와 동일하도록 설정
    updateTarget(targetOps, sess)
    for i in range(total_episodes):
        episodeBuffer = experience_buffer()
        # 환경을 리셋하고 새로운 관찰을 얻는다.
        s = env.reset()
        s = processState(s)
        done = False
        TotalReward = 0
        step = 0

        # 1. Q 네트워크
        # 최대 50회 에이전트가 블록에 도달하기까지 시도하고 종료
        while step < max_epLength:
            step+=1
            # Q 네트워크에서 e의 확률로 랜덤한 액션을 취하거나 그리디하게 액션을 선택.
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                action = np.random.randint(0, 4) # 0, 1, 2, 3
            else:
                action = sess.run(main_Q_Network.predict, feed_dict={main_Q_Network.scalarInput:[s]})[0]
            s_, r, done = env.step(action)
            s_ = processState(s_)

            # 에피소드별 단계수 저장을 위한 total_steps (한 에피소드 내)
            total_steps += 1
            # 에피소드 버퍼에 경험 저장
            episodeBuffer.add(np.reshape(np.array([s, action, r, s_, done]), [1, 5]))

            if total_steps > pre_train_steps:
                if e > e_end:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    # 경험에서 랜덤하게 배치 하나 샘플링
                    trainBatch = myBuffer.sample(batch_size)
                    # 타겟 Q 값에 대해 더블 DQN 업데이트
                    Q1 = sess.run(main_Q_Network.predict,\
                                  feed_dict={main_Q_Network.scalarInput:np.vstack(trainBatch[:,3])})
                    Q2 = sess.run(target_Q_Network.Qout,\
                                  feed_dict={target_Q_Network.scalarInput:np.vstack(trainBatch[:,3])})
                    end_multiplier = -(trainBatch[:,4] -1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (discount_factor*doubleQ * end_multiplier)

                    # 타겟 값을 활용해 업데이트
                    _ = sess.run(main_Q_Network.updatemodel,\
                                 feed_dict={main_Q_Network.scalarInput:np.vstack(trainBatch[:,0]),\
                                            main_Q_Network.targetQ: targetQ, main_Q_Network.actions:trainBatch[:,1]})
                #타겟 네트워크가 제 1네트워크와 동일하게 설정
                updateTarget(targetOps, sess)
            TotalReward += r
            s = s_

            if done == True:
                break
        myBuffer.add(episodeBuffer.buffer)
        stepList.append(step)
        RList.append(TotalReward)
        #정기적으로 모델 저장
        if i % 1000 == 0:
            saver.save(sess, path+'/model-'+str(i)+'.ckpt')
            print("saved model")
        if len(RList) % 10 == 0:
            print(total_steps, np.mean(RList[-10:]), e)
print("Percent of successful episodes: " + str(sum(RList)/total_episodes))