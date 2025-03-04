import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      # rank_num * tensor_size

def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
def ndf_loss(image_feats, text_feats, labels, logit_scale, world=False):
    if world:
        image_feats = all_gather_with_grad(image_feats)
        text_feats = all_gather_with_grad(text_feats)
        labels = concat_all_gather(labels)
    bs = image_feats.shape[0]
    epsilon = 1e-8
    labels = labels.view(-1,bs)
    labels = torch.eq(labels, labels.t()).float()
    labels = labels / labels.sum(dim=1)

    # normalized features
    image_norm = image_feats / image_feats.norm(dim=-1, keepdim=True)
    text_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t() #bs x bs

    logits_per_text = logits_per_image.t()

    loss_i = F.softmax(logits_per_image,dim=1) * (F.log_softmax(logits_per_image, dim=1) - torch.log(labels + epsilon)) + \
                labels * (torch.log(labels + epsilon) - F.log_softmax(logits_per_image, dim=1)) + F.cross_entropy(logits_per_image, labels)
    loss_t = F.softmax(logits_per_text,dim=1) * (F.log_softmax(logits_per_text, dim=1) - torch.log(labels + epsilon)) + \
                 labels * (torch.log(labels + epsilon) - F.log_softmax(logits_per_text, dim=1)) + F.cross_entropy(logits_per_text, labels)
    loss = (loss_i.mean() + loss_t.mean()) / 2


    return loss

def compute_TRL_per(image_features, text_features, pid, margin = 0.2, tau=0.02):

    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask*scores).max(1)[0]
    neg_2 = (mask*scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)

    loss = (cost_1 + cost_2).sum()

    return loss

 
def compute_InfoNCE_per(image_features, text_features, logit_scale):

    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2 
    
    return loss


def compute_TAL_per(image_features, text_features, pid, tau, margin):

    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1,
                                                                                             keepdim=True)).detach()

    loss = (-  (alpha_i2t * scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0) \
           + (-  (alpha_t2i * scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0)

    loss = loss.sum()

    return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1))
        pt = pt.gather(1, target.unsqueeze(1))

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.to(input.device)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at.view(-1, 1)

        loss = -1 * (1 - pt)**self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
def compute_mlm(self, scores, labels):

    fl = FocalLoss(gamma=2, ignore_index=0)
    return fl(scores.view(-1, scores.size(-1)), labels.view(-1))

    #ce = nn.CrossEntropyLoss(ignore_index=0)
    #return ce(scores, labels)

def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2