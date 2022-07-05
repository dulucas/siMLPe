import torch

class H36MHuman(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self._some_variables()

    def _some_variables(self):

        parent = torch.tensor([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1
        self.parent = parent.to(self.device).long()

        offset = torch.tensor(
            [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
             -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
             0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
             0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
             257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
             0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
             0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
             0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
             0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
             0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000]).float()
        self.offset = offset.reshape(-1, 3).to(self.device)

    def forward(self, rotmat):
        assert rotmat.shape[1] == 32
        n = rotmat.data.shape[0]
        j_n = self.offset.shape[0]
        p3d = self.offset.clone().unsqueeze(0).repeat(n, 1, 1)
        p3d_ = [p3d[:,0,:].clone()] #self.offset.clone().unsqueeze(0).repeat(n, 1, 1)
        R = rotmat.reshape(n, j_n, 3, 3).clone()
        R_ = [R[:,0,:,:].clone()] #rotmat.reshape(n, j_n, 3, 3).clone()
        for i in range(1, j_n):
            if self.parent[i] > 0:
                R_.append(torch.matmul(R[:, i, :, :], R_[self.parent[i]]))
                p3d_.append(torch.matmul(p3d[0, i, :], R_[self.parent[i]]) + p3d_[self.parent[i]])
            else:
                R_.append(R[:,i,:,:].clone())
                p3d_.append(p3d[:,i,:])
        p3d_ = torch.stack(p3d_).permute(1,0,2)
        return p3d_


