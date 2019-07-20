
import torch.nn as nn
import torch


class MSESequenceLoss(nn.Module):
    # def __init__(self):
    #     super(MSESequenceLoss, self).__init__()
    #
    # def forward(self, inputs, targets):
    #     T = inputs.shape[1]
    #     if targets.shape[1] != T:
    #         f_0 = torch.unsqueeze(targets[:, 0, :, :, :], 1)
    #         targets = torch.cat([f_0, targets], dim=1)
    #     return torch.mean(inputs.sub(targets) ** 2)


    def __init__(self):
        super(MSESequenceLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)


    def forward(self, output, labels):
        batch_size = output.size(0)
        num_stages = output.size(1)
        num_joints = output.size(2)
        T = output.shape[1]
        if labels.shape[1] != T:
            f_0 = torch.unsqueeze(labels[:, 0, :, :, :], 1)
            labels = torch.cat([f_0, labels], dim=1)
        loss = 0

        for i_stage in range(0,num_stages):
            heatmaps_pred = output[:,i_stage,:,:,:].reshape((batch_size,num_joints, -1)).split(1, 1)
            heatmaps_gt = labels[:,i_stage,:,:,:].reshape((batch_size, num_joints, -1)).split(1, 1)


            for idx in range(num_joints):
                heatmap_pred = heatmaps_pred[idx].squeeze()
                #remove dimention = 1
                heatmap_gt = heatmaps_gt[idx].squeeze()

                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / (num_joints)*num_stages

