import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GRPOLoss(nn.module):
    def __init__(self, 
                 clip_param=0.2, 
                 value_loss_coef=0.5, 
                 entropy_coef=0.01):
        super(GRPOLoss, self).__init__()
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
    
    def forward(self, 
                old_log_probs, 
                new_log_probs, 
                advantages, 
                returns, 
                values, 
                entropy):
        
        # compute policy ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # surrogate loss clipping
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # value loss
        value_loss = (returns - values).pow(2).mean()

        # total loss (w/ negative entropy bonus to encourage exploration)
        total_loss = policy_loss + (self.value_loss_coef * value_loss) - (self.entropy_coef * entropy.mean())

        return total_loss

def compute_group_relative_advantages(advantages, group_ids):
    """
    for each sample in the batch, subtract the mean advantage of its group.
    this centers advantages relative to the group, the key to GRPO.
    """
    # clone to not disturb original advantages
    relative_advantages = advantages.clone()

    # get unique groups
    unique_groups = torch.unique(group_ids)

    # iterate over groups
    for group in unique_groups:
        # mask for group
        mask = (group_ids == group)

        # compute mean advantage of group
        group_mean = advantages[mask].mean()

        # subtract mean from advantages
        relative_advantages = advantages[mask] - group_mean
    
    # return relative advantages
    return relative_advantages