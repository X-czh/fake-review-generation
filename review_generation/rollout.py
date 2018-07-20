import copy
import numpy as np

import torch
import torch.nn as nn


def redistribution(idx, total, min_v):
    idx = (idx + 0.0) / (total + 0.0) * 16.0
    return (np.exp(idx - 8.0) / (1.0 + np.exp(idx - 8.0)))


def rescale(reward, num):
    reward = np.array(reward)
    x, y = reward.shape
    ret = np.zeros((x, y))
    for i in range(x):
        l = reward[i]
        rescalar = {}
        for s in l:
            rescalar[s] = s
        idxx = 1
        min_s = 1.0
        max_s = 0.0
        for s in rescalar:
            rescalar[s] = redistribution(idxx, len(l), min_s)
            idxx += 1
        for j in range(y):
            ret[i, j] = rescalar[reward[i, j]]
    return ret


class Rollout(object):
    """ Rollout Policy """

    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, x, num, discriminator):
        """
        Inputs: x, num, discriminator
            - x: (batch_size, seq_len) input data
            - num: rollout number
            - discriminator: discriminator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                data = x[:, 0:l]
                samples = self.own_model.sample(batch_size, seq_len, data)
                pred = discriminator(samples)
                pred = pred.cpu().data[:,1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = discriminator(x)
            pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        rewards = rescale(np.array(rewards), num) # Bootstrapped Rescaled Activation
        rewards = np.transpose(np.array(rewards, dtype=np.float32)) / (1.0 * num) # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
