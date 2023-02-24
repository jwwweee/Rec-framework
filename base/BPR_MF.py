import torch
import torch.nn as nn

class BPR_MF(nn.Module):
    def __init__(self, args):
        super(BPR_MF, self).__init__()
        
        self.batch_size = args.batch_size
        self.reg_coef = eval(args.regs)[0]

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
