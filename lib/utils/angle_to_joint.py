import torch

def ang2joint(p3d0, pose,
              parent={0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9,
                      15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}):
    """

    :param p3d0:[batch_size, joint_num, 3]
    :param pose:[batch_size, joint_num, 3]
    :param parent:
    :return:
    """
    batch_num = p3d0.shape[0]
    jnum = len(parent.keys())
    J = p3d0
    R_cube_big = rodrigues(pose.contiguous().view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)
    results = []
    results.append(
        with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
    )
    for i in range(1, jnum):
        results.append(
            torch.matmul(
                results[parent[i]],
                with_zeros(
                    torch.cat(
                        (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))),
                        dim=2
                    )
                )
            )
        )

    stacked = torch.stack(results, dim=1)
    J_transformed = stacked[:, :, :3, 3]
    return J_transformed

def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size * angle_num, 3, 3].

    """
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
    # theta = torch.norm(r, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float).to(r.device)
    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
         -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0) \
              + torch.zeros((theta_dim, 3, 3), dtype=torch.float)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R

def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

    Parameter:
    ---------
    x: Tensor to be appended.

    Return:
    ------
    Tensor after appending of shape [4,4]

    """
    ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float
    ).expand(x.shape[0], -1, -1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret


