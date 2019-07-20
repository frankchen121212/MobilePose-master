
import torch
import numpy as np
import math
def accuracy(inputs, targets, visibility ,r=0.2):
    batch_size = inputs.shape[0]
    n_stages = inputs.shape[1]
    if visibility.shape[1] != n_stages:
        #the dataset is Image dataset #LSP or MPII
        visibility_tmp = np.ones((batch_size,n_stages,visibility.shape[1]))
        for i_stage in range (n_stages):
            for i_batch in range (batch_size):
                for i_joint in range (visibility.shape[1]):
                    visibility_tmp[i_batch,i_stage,i_joint]  = visibility[i_batch,i_joint]
        visibility = torch.Tensor(visibility_tmp)
    n_joints = visibility.shape[2]
    inputs = inputs.detach()
    targets = targets.detach()
    
    if targets.shape[1] != n_stages:
        f_0 = torch.unsqueeze(targets[:, 0, :, :, :], 1)
        targets = torch.cat([f_0, targets], dim=1)
    n_correct = 0
    n_total = 0

    # names = ['ra'0, 'rk'1, 'rh'2, 'lh'3, 'lk'4, 'la'5, 'le'6, 'lw'7,
    #  'neck'8, 'head'9, 'rw'10, 're'11, 'rs'12, 'ls'13]
    eval_dic = np.zeros((7,1))
    #Head #Sho #Elb #Wri #Hip #Knee #Ank
    n_head, n_sholder, n_elbow, n_wrist, n_hip, n_knee, n_ankle = 0, 0, 0, 0, 0, 0, 0
    n_head_correct, n_sholder_correct, n_elbow_correct, n_wrist_correct,\
    n_hip_correct, n_knee_correct, n_ankle_correct = 0, 0, 0, 0, 0, 0, 0
    for i in range(batch_size):
        gt = get_preds(targets[i, :, :, :, :])
        preds = get_preds(inputs[i, :, :, :, :])
        batch_vis = visibility[i,:,:]
        for j in range(n_stages):
            w = gt[j, :, 0].max() - gt[j, :, 0].min()
            h = gt[j, :, 1].max() - gt[j, :, 1].min()
            threshold = r * max(w, h)
            stage_vis = batch_vis[j,:]
            for i_joint in range(n_joints):
                gt_joint = gt[j][i_joint].numpy()
                pred_joint = preds[j][i_joint].numpy()
                
                # the gt_joint is invisible
                # do not count in the total joints
                # TODO only count the visible ones
                if stage_vis[i_joint]==0:
                    continue
                # TODO ignore the neck
                if i_joint == 8:
                    continue

                error_dist = np.sqrt((gt_joint[0]-pred_joint[0])**2 + (gt_joint[1]-pred_joint[1])**2)
                hit = error_dist <=threshold

                if hit:
                    n_correct+=1
                    if i_joint==9:
                        n_head_correct +=1
                        n_head+=1

                    if i_joint==12 or i_joint==13:
                        n_sholder_correct +=1
                        n_sholder+=1

                    if i_joint == 6 or i_joint == 11:
                        n_elbow_correct+=1
                        n_elbow+=1

                    if i_joint==7 or i_joint==10:
                        n_wrist_correct+=1
                        n_wrist+=1

                    if i_joint==2 or i_joint==3:
                        n_hip_correct+=1
                        n_hip+=1

                    if i_joint==1 or i_joint==4:
                        n_knee_correct+=1
                        n_knee+=1

                    if i_joint==0 or i_joint==5:
                        n_ankle_correct+=1
                        n_ankle+=1
                else:
                    if i_joint == 9:
                        n_head += 1

                    if i_joint == 12 or i_joint == 13:
                        n_sholder += 1

                    if i_joint == 6 or i_joint == 11:
                        n_elbow += 1

                    if i_joint == 7 or i_joint == 10:
                        n_wrist += 1

                    if i_joint == 2 or i_joint == 3:
                        n_hip += 1

                    if i_joint == 1 or i_joint == 4:
                        n_knee += 1

                    if i_joint == 0 or i_joint == 5:
                        n_ankle += 1

                n_total +=1
    if n_head!=0:
        eval_dic[0] = n_head_correct / n_head
    if n_sholder != 0:
        eval_dic[1] = n_sholder_correct / n_sholder
    if n_elbow!=0:
        eval_dic[2] = n_elbow_correct / n_elbow
    if n_wrist!=0:
        eval_dic[3] = n_wrist_correct / n_wrist
    if n_hip !=0:
        eval_dic[4] = n_hip_correct / n_hip
    if n_knee!=0:
        eval_dic[5] = n_knee_correct / n_knee
    if n_ankle!=0:
        eval_dic[6] = n_ankle_correct / n_ankle


    return float(n_correct) / float(n_total) ,eval_dic




def coord_accuracy(inputs, gt, r=0.2):
    batch_size = inputs.shape[0]
    n_stages = inputs.shape[1]
    n_joints = inputs.shape[2]

    inputs = inputs.detach()
    gt = gt.detach()

    n_correct = 0
    n_total = batch_size * n_stages * n_joints

    for i in range(batch_size):
        w = gt[i, :, 0].max() - gt[i, :, 0].min()
        h = gt[i, :, 1].max() - gt[i, :, 1].min()
        threshold = r * max(w, h)

        curr_gt = torch.unsqueeze(gt[i], 0).repeat(n_stages, 1, 1)
        scores = torch.norm(inputs[i].sub(curr_gt), dim=2).view(-1)
        n_correct += scores.le(threshold).sum()

    return float(n_correct) / float(n_total)


# Source: https://github.com/bearpaw/pytorch-pose/blob/master/pose/utils/evaluation.py
def get_preds(scores):

    batch_heatmaps = scores.cpu().numpy()

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask

    coords = preds.copy()
    heatmap_height = scores.shape[2]
    heatmap_width = scores.shape[3]
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = scores[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]])
                coords[n][p] = np.sign(diff) * 0.25 + coords[n][p]

    preds = torch.from_numpy(coords).float()

    return preds
