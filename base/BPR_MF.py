import torch
import torch.nn as nn

class BPR_MF(nn.Module):
    def __init__(self, args):
        super(BPR_MF, self).__init__()
        
        self.batch_size = args.batch_size
        self.reg_coef = eval(args.regs)[0]

        self = self.to(self.device)
    
    def loss_func(self, user_g_embeddings, pos_item_g_embeddings, neg_item_g_embeddings):
        """ BPR loss function, compute BPR loss for ranking task in recommendation.
        """

        # compute positive and negative scores
        pos_scores = torch.sum(torch.mul(user_g_embeddings, pos_item_g_embeddings), axis=1)
        neg_scores = torch.sum(torch.mul(user_g_embeddings, neg_item_g_embeddings), axis=1)
        
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # compute regularizer
        regularizer = (torch.norm(user_g_embeddings) ** 2
                       + torch.norm(pos_item_g_embeddings) ** 2
                       + torch.norm(neg_item_g_embeddings) ** 2) / 2

        for param in self.parameters():
            regularizer += torch.sum(torch.square(param))

        emb_loss = self.reg_coef * regularizer / self.batch_size
        
        batch_loss = mf_loss + emb_loss

        return batch_loss

    def predict_score(self, user_g_embeddings, all_item_g_embeddings):
        """ Predict the score of a pair of user-item interaction
        """
        score = torch.matmul(user_g_embeddings, all_item_g_embeddings.t())

        return score
        
    def _convert_sp_mat_to_sp_tensor(self, L):
        """ Convert sparse mat to sparse tensor.
        """
        coo = L.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(indices, values, coo.shape)
