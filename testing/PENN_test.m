gt_root= ('/share1/home/siyuan/MobilePose-master/output/lpm/PennAction/test/gt.mat');
pred_root= ('/share1/home/siyuan/MobilePose-master/output/lpm/PennAction/test/pred.mat');
vis_root= ('/share1/home/siyuan/MobilePose-master/output/lpm/PennAction/test/vis.mat');
bbox_root = ('/share1/home/siyuan/MobilePose-master/output/lpm/PennAction/test/box.mat');

gt= cell2mat(struct2cell(load(gt_root)));
pred= cell2mat(struct2cell(load(pred_root)));
vis= cell2mat(struct2cell(load(vis_root)));
bbox = cell2mat(struct2cell(load(bbox_root)));
numSeq = length(pred);

numSeq_bbox = length(bbox);
fprintf('testing %d samples ==> ',numSeq_bbox);

orderToPENN = [1 3 6 4 7 5 8 9 12 10 13 11 14]; %ignore the second one which is the neck
torso_norm = 0; %1:Torso / 0:bbox; default as 0 -> 0.2*max(h,w)

obj = zeros(1,length(orderToPENN));
detected = zeros(1,length(orderToPENN));


for i = 1:numSeq
    fprintf('%d / %d \n',i,numSeq);
    gt_coord = gt(i,:,:);
    pred_coord = pred(i,:,:);
    vis_coord = vis(i,:,:);
    bbox_coord = bbox(i,:,:);
    gt_x = gt_coord(:,:,1);
    gt_y = gt_coord(:,:,2);
    pred_x = pred_coord(:,:,1);
    pred_y = pred_coord(:,:,2);
    
    %compute bodysize
    min_x = min(min(gt_x));
    max_x = max(max(gt_x));
    min_y = min(min(gt_y));
    max_y = max(max(gt_y));
    w = max_x - min_x;
    h = max_y - min_y;
    if(torso_norm == 1)
        bodysize = max(w,h);
    else
    bodysize = max(max(bbox_coord(:,3)-bbox_coord(:,1),bbox_coord(:,4)-bbox_coord(:,2)));
    end
    
    for joints = 1:length(orderToPENN)
        hit=2;
        if(vis_coord(orderToPENN(joints))==1)
            error_dist = norm([pred_x(orderToPENN(joints)),pred_y(orderToPENN(joints))] - [gt_x(orderToPENN(joints)),gt_y(orderToPENN(joints))]);
            hit = error_dist <= bodysize*0.2;
            obj(joints) = obj(joints) + 1;
            if(hit)
               detected(joints) = detected(joints) + 1;
            end
        end
        fprintf(' %d', hit);
    end
    
    fprintf(' |');
    
    for joints = 1:length(orderToPENN)
        if(obj(joints)==0)
            fprintf(' 0');
        else
            fprintf(' %.3f', detected(joints)/obj(joints));
        end
    end
    if(sum(obj)==0)
        fprintf(' ||0\n');
        acc = 0;
    else
        acc = sum(detected)/sum(obj);
        fprintf(' ||%.4f\n',acc );
    end

    
end
exit()



