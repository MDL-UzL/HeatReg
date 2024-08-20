import torch
import torch.nn as nn
import torch.nn.functional as F

'''
FreePointTransformer and HeatRegNet models.
'''

class FreePointTransformer(torch.nn.Module):
    def __init__(self, base=16):
        super(FreePointTransformer, self).__init__()
        
        self.global_feat = nn.Sequential(nn.Conv1d(3, base, 1), nn.GroupNorm(4, base), nn.ReLU(),
                                         nn.Conv1d(base, base, 1), nn.GroupNorm(4, base), nn.ReLU(),
                                         nn.Conv1d(base, base, 1), nn.GroupNorm(4, base), nn.ReLU(),
                                         nn.Conv1d(base, 2*base, 1), nn.GroupNorm(4, 2*base), nn.ReLU(),
                                         nn.Conv1d(2*base, 16*base, 1), nn.GroupNorm(4, 16*base), nn.ReLU())
        
        self.disp = nn.Sequential(nn.Conv1d(3+16*base+16*base, 16*base, 1), nn.GroupNorm(4, 16*base), nn.ReLU(),
                                  nn.Conv1d(16*base, 8*base, 1), nn.GroupNorm(4, 8*base), nn.ReLU(),
                                  nn.Conv1d(8*base, 3, 1))
            
    def forward(self, kpts_fixed, kpts_moving):
        B, N, _ = kpts_fixed.shape
        
        global_fixed = self.global_feat(kpts_fixed.permute(0, 2, 1)).max(2, keepdim=True)[0]
        global_moving = self.global_feat(kpts_moving.permute(0, 2, 1)).max(2, keepdim=True)[0]
                
        disp = self.disp(torch.cat([kpts_fixed.permute(0, 2, 1), global_fixed.expand(-1, -1, N), global_moving.expand(-1, -1, N)], 1))
        
        return disp.permute(0, 2, 1)
    
class HeatRegNet(torch.nn.Module):
    def __init__(self, k, stride=1, base=16):
        super(HeatRegNet, self).__init__()
        self.k = k
        self.stride = stride
        
        self.global_feat = nn.Sequential(nn.Conv1d(3, base, 1), nn.GroupNorm(4, base), nn.ReLU(),
                                         nn.Conv1d(base, base, 1), nn.GroupNorm(4, base), nn.ReLU(),
                                         nn.Conv1d(base, base, 1), nn.GroupNorm(4, base), nn.ReLU(),
                                         nn.Conv1d(base, 2*base, 1), nn.GroupNorm(4, 2*base), nn.ReLU(),
                                         nn.Conv1d(2*base, 16*base, 1), nn.GroupNorm(4, 16*base), nn.ReLU())
        
        self.disp = nn.Sequential(nn.Conv2d(3+3+16*base+16*base, 16*base, 1), nn.GroupNorm(4, 16*base), nn.ReLU(),
                                  nn.Conv2d(16*base, 8*base, 1), nn.GroupNorm(4, 8*base), nn.ReLU(),
                                  nn.Conv2d(8*base, 1, 1))
    
    def forward(self, kpts_fixed, kpts_moving):
        B, N, _ = kpts_fixed.shape
        
        global_fixed = self.global_feat(kpts_fixed.permute(0, 2, 1)).max(2, keepdim=True)[0]
        global_moving = self.global_feat(kpts_moving.permute(0, 2, 1)).max(2, keepdim=True)[0]
        
        dist = torch.cdist(kpts_fixed, kpts_moving)
        ind = (-dist).topk(self.k*self.stride, dim=-1)[1][:, :, ::self.stride]
        candidates = - kpts_fixed.view(B, N, 1, 3) + kpts_moving.gather(1, ind.view(B, -1, 1).expand(-1, -1, 3)).view(B, N, self.k, 3)
        
        disp = self.disp(torch.cat([kpts_fixed.view(B, N, 1, 3).expand(-1, -1, self.k, -1).permute(0, 3, 1, 2),
                                    candidates.permute(0, 3, 1, 2),
                                    global_fixed.view(B, -1, 1, 1).expand(-1, -1, N, self.k),
                                    global_moving.view(B, -1, 1, 1).expand(-1, -1, N, self.k)], 1))
            
        return (candidates.permute(0, 3, 1, 2) * F.softmax(disp, 3)).sum(-1).permute(0, 2, 1)