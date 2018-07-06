import torch
import numpy as np


class Rollout:
    """
    Rollout policy
    """
    
    def __init__(self, model):
        self.model = model

    def get_rewards(self, x, num_rollout, discriminator):
        """
        Inputs: x, num_rollout, discriminator
            - x: (batch_size, seq_len), input data
            - num_rollout: rollout number
            - discriminator: discriminator model
        Outputs: rewards
            - rewards: (batch_size, seq_len)
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num_rollout):
            for l in range(1, seq_len):
                data = x[:, 0:l]
                samples = self.model.sample(batch_size, seq_len, data)
                pred = discriminator(samples)
                pred = pred.cpu().data[:, 1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l - 1] += pred

            # last token reward
            pred = discriminator(x)
            pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len - 1] += pred

        rewards = np.transpose(np.array(rewards)) / (1.0 * num_rollout)  # batch_size * seq_length
        return rewards
