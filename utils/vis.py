
import cv2
import os
import torchvision
import numpy as np
# def save_mapped_batches():
#
#
# def save_meta_batches():
#
# def save_raw_batches():

def save_visualize_result(batch_image,labels,batch_labels,raw_image,flipped_labels,x_raw,
                          y_raw,v_raw,vis_flipped,output_root,i_loader,j_batch):
    # batch_image.shape ([5, 3, 256, 256])
    # labels.shape ([1, 5, 15, 31, 31])
    # batch_labels.shape ([5,15,2])
    # raw_image.shape ([ 5, 3 , width_raw,height_raw ])
    # flipped_labels.shape ([1,5,28])[x1,x2,x3 ...,x14,y1,y2,y3...y14]
    # x_raw,y_raw,v_raw,[0, 5, 13]
    # vis_flipped [0, 5, 14]
    # i_loader -- which loader, j_batch -- which_batch


    batch_size, n_stages, n_joints = labels.shape[0], labels.shape[1], labels.shape[2]
    xmaps = n_stages
    ymaps = batch_size

    image_size = batch_image.shape[-2]
    label_size = labels.shape[-2]

    width_raw, height_raw = (raw_image.shape[2], raw_image.shape[3])
    rotation = image_size / label_size

    grid = torchvision.utils.make_grid(batch_image, nrow=n_stages, padding=2, normalize=True)
    ndarr = grid.mul(255).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
    raw_grid = torchvision.utils.make_grid(raw_image, nrow=n_stages, padding=2, normalize=True)
    raw_ndarr = raw_grid.mul(255).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
    b, g, r = cv2.split(raw_ndarr)
    raw_ndarr = cv2.merge([r, g, b])

    b, g, r = cv2.split(ndarr)
    ndarr = cv2.merge([r, g, b])
    ndarr = ndarr.copy()
    ndarr_meta = ndarr.copy()
    padding = 2

    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    # mpii_order = [13, 11, 9, 8, 10, 12, 4, 6, 14, 1, 7, 5, 3, 2]
    # transformed order [13, 11,  9,  8, 10, 12,  4,  6, 14,  1,  7,  5,  3,  2]
    names = ['ra', 'rk', 'rh', 'lh', 'lk', 'la', 'le', 'lw', 'neck', 'head', 'rw', 're', 'rs', 'ls']

    ### raw ####
    for y in range(ymaps):
        for x in range(xmaps):
            raw_xs = x_raw[0, k, :]
            raw_ys = y_raw[0, k, :]
            raw_vis = v_raw[0, k, :]
            # i_name = 0
            for i_name, (raw_x, raw_y, raw_vi) in enumerate(zip(raw_xs, raw_ys, raw_vis)):
                if raw_vi == 0:
                    continue
                raw_x = x * height_raw + padding + raw_x
                raw_y = y * width_raw + padding + raw_y
                cv2.circle(raw_ndarr, (int(raw_x), int(raw_y)), 2, [0, 255, 0], 2)
                cv2.putText(raw_ndarr, names[i_name], org=(int(raw_x), int(raw_y)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=0.5, color=[0, 0, 255])
                # i_name +=1
            k = k + 1
    cv2.imwrite(os.path.join(output_root, 'loader_' + str(i_loader) + '_batch_' + str(j_batch) + '_raw.png'),
                raw_ndarr)
    print('loader_' + str(i_loader) + '_batch_' + str(j_batch) + '_raw.png' + 'saved successfuly!')

    ### mapped ###
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            raw_vis = vis_flipped[0, k, :]
            joints = batch_labels[k, :, :] * rotation
            for i_name, joint in enumerate(joints):
                if i_name < 14:
                    if raw_vis[i_name] == 0:
                        continue
                    joint[0] = x * width + padding + joint[0]
                    joint[1] = y * height + padding + joint[1]
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
                    cv2.putText(ndarr, names[i_name], org=(int(joint[0]), int(joint[1])),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=[0, 0, 255])
            k = k + 1
    cv2.imwrite(os.path.join(output_root, 'loader_' + str(i_loader) + '_batch_' + str(j_batch) + '_mapped.png'), ndarr)
    print('loader_' + str(i_loader) + '_batch_' + str(j_batch) + '_mapped.png' + 'saved successfuly!')

    ### meta ###
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            flipped_xs = flipped_labels[0, k, :][0:14]
            flipped_ys = flipped_labels[0, k, :][14:]
            fipped_vis = vis_flipped[0, k, :]
            for i_flipped, (flipped_x, flipped_y) in enumerate(zip(flipped_xs, flipped_ys)):
                if fipped_vis[i_flipped] == 0:
                    continue
                flipped_x = x * width + padding + flipped_x
                flipped_y = y * height + padding + flipped_y
                cv2.circle(ndarr_meta, (int(flipped_x), int(flipped_y)), 2, [0, 0, 255], 2)
                cv2.putText(ndarr_meta, names[i_flipped], org=(int(flipped_x), int(flipped_y)),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=[0, 0, 255])
            k = k + 1
    cv2.imwrite(os.path.join(output_root, 'loader_' + str(i_loader) + '_batch_' + str(j_batch) + '_transformed.png'), ndarr_meta)
    print('loader_' + str(i_loader) + '_batch_' + str(j_batch) + '_transformed.png' + 'saved successfuly!')

    ### heatmap ###
    # batch_image.shape ([5, 3, 256, 256])
    # labels.shape ([1, 5, 15, 31, 31])
    # heatmap.shape ([ 5, 15, 31, 31])
    # batch_labels.shape ([5,15,2])

    # stage_heatmap = labels[0].clone().cpu()
    # stage_size = batch_image.size(0)
    # num_joints = stage_heatmap.size(1)
    # image_height = int(batch_image.size(2))
    # image_width = int(batch_image.size(3))
    # grid_image = np.zeros((stage_size*image_height,
    #                        (num_joints+1)*image_width,
    #                        3),
    #                       dtype=np.float32
    #                       )
    # for y in range(stage_size):
    #     image = batch_image[y].mul(255).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
    #     b, g, r = cv2.split(image)
    #     image = cv2.merge([r, g, b])
    #
    #     heatmaps = stage_heatmap[y].mul(255) \
    #         .clamp(0, 255) \
    #         .byte() \
    #         .cpu().numpy()
    #
    #     height_begin = image_height * y
    #     height_end = image_width * (y + 1)
    #
    #     for x in range(num_joints):
    #         cv2.circle(image,
    #                    (int(batch_labels[y][x][0]), int(batch_labels[y][x][1])),
    #                    1, [0, 0, 255], 1)
    #         heatmap = heatmaps[y, :, :]
    #         heatmap_resized = cv2.resize(heatmap,(image_width,image_height))
    #         colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    #         masked_image = colored_heatmap * 0.2 + image*0.7
    #         cv2.circle(masked_image,
    #                    (int(batch_labels[y][x][0]), int(batch_labels[y][x][1])),
    #                    1, [0, 0, 255], 1)
    #
    #         width_begin = image_height * (x + 1)
    #         width_end = image_width * (x + 2)
    #         grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image
    #         # grid_image[height_begin:height_end, width_begin:width_end, :] = \
    #         #     colored_heatmap*0.7 + resized_image*0.3
    #
    #     grid_image[height_begin:height_end, 0:image_width, :] = image
    #
    # cv2.imwrite(os.path.join(output_root, 'loader_' + str(i_loader) + '_batch_' + str(j_batch) + '_heatmap.png'), grid_image)
    #
    # # cv2.imwrite(os.path.join(output_root, 'loader_' + str(i_loader) + '_batch_' + str(j_batch) + '_heatmap.png'), heatmap_ndarr)
    # print('loader_' + str(i_loader) + '_batch_' + str(j_batch) + '_heatmap.png' + 'saved successfuly!')