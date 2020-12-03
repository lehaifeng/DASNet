import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.nn import functional as F
import utils.transforms as trans
import utils.utils as util
import utils.metric as mc
import time
import datetime
import cv2
import dataset.rs as dates

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def various_distance(out_vec_t0, out_vec_t1,dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance

def single_layer_similar_heatmap_visual(output_t0,output_t1,save_change_map_dir,epoch,filename,layer_flag,dist_flag):

    fname = filename[7:12]
    n, c, h, w = output_t0.data.shape
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    similar_distance_map_rz = nn.functional.interpolate(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]),size=[256,256], mode='bilinear',align_corners=True)
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    save_change_map_dir_ = os.path.join(save_change_map_dir, 'epoch_' + str(epoch))
    check_dir(save_change_map_dir_)
    save_change_map_dir_layer = os.path.join(save_change_map_dir_,layer_flag)
    check_dir(save_change_map_dir_layer)
    save_weight_fig_dir = os.path.join(save_change_map_dir_layer, fname + '.jpg')
    cv2.imwrite(save_weight_fig_dir, similar_dis_map_colorize)
    return similar_distance_map_rz.data.cpu().numpy()



def validate(net, val_dataloader,save_change_map_dir,save_roc_dir):
    epoch = 1
    net.eval()
    with torch.no_grad():
        cont_conv5_total, cont_fc_total, cont_embedding_total, num = 0.0, 0.0, 0.0, 0.0
        metric_for_conditions = util.init_metric_for_class_for_cmu(1)
        for batch_idx, batch in enumerate(val_dataloader):
            inputs1, input2, targets, filename, height, width = batch
            height, width, filename = height.numpy()[0], width.numpy()[0], filename[0]
            inputs1, input2, targets = inputs1.cuda(), input2.cuda(), targets.cuda()
            fname = filename.split('/')[1][:-4]
            out_conv5, out_fc, out_embedding = net(inputs1, input2)
            out_conv5_t0, out_conv5_t1 = out_conv5
            out_fc_t0, out_fc_t1 = out_fc
            out_embedding_t0, out_embedding_t1 = out_embedding
            conv5_distance_map = single_layer_similar_heatmap_visual(out_conv5_t0, out_conv5_t1, save_change_map_dir,
                                                                     epoch, filename, 'conv5', 'l2')
            fc_distance_map = single_layer_similar_heatmap_visual(out_fc_t0, out_fc_t1, save_change_map_dir, epoch,
                                                                  filename, 'fc', 'l2')
            embedding_distance_map = single_layer_similar_heatmap_visual(out_embedding_t0, out_embedding_t1,
                                                                         save_change_map_dir, epoch, filename,
                                                                         'embedding', 'l2')
            cont_conv5 = mc.RMS_Contrast(conv5_distance_map)
            cont_fc = mc.RMS_Contrast(fc_distance_map)
            cont_embedding = mc.RMS_Contrast(embedding_distance_map)
            cont_conv5_total += cont_conv5
            cont_fc_total += cont_fc
            cont_embedding_total += cont_embedding
            num += 1
            prob_change = embedding_distance_map[0][0]
            gt = targets.data.cpu().numpy()
            FN, FP, posNum, negNum = mc.eval_image_rewrite(gt[0], prob_change, cl_index=1)
            metric_for_conditions[0]['total_fp'] += FP
            metric_for_conditions[0]['total_fn'] += FN
            metric_for_conditions[0]['total_posnum'] += posNum
            metric_for_conditions[0]['total_negnum'] += negNum
            cont_conv5_mean, cont_fc_mean, cont_embedding_mean = cont_conv5_total / num, \
                                                                 cont_fc_total / num, cont_embedding_total / num

        thresh = np.array(range(0, 256)) / 255.0
        conds = metric_for_conditions.keys()
        for cond_name in conds:
            total_posnum = metric_for_conditions[cond_name]['total_posnum']
            total_negnum = metric_for_conditions[cond_name]['total_negnum']
            total_fn = metric_for_conditions[cond_name]['total_fn']
            total_fp = metric_for_conditions[cond_name]['total_fp']
            metric_dict = mc.pxEval_maximizeFMeasure(total_posnum, total_negnum,
                                                     total_fn, total_fp, thresh=thresh)
            metric_for_conditions[cond_name].setdefault('metric', metric_dict)
            metric_for_conditions[cond_name].setdefault('contrast_conv5', cont_conv5_mean)
            metric_for_conditions[cond_name].setdefault('contrast_fc', cont_fc_mean)
            metric_for_conditions[cond_name].setdefault('contrast_embedding', cont_embedding_mean)

        f_score_total = 0.0
        for cond_name in conds:
            pr, recall, f_score = metric_for_conditions[cond_name]['metric']['precision'], \
                                  metric_for_conditions[cond_name]['metric']['recall'], \
                                  metric_for_conditions[cond_name]['metric']['MaxF']
            roc_save_epoch_dir = os.path.join(save_roc_dir, str(epoch))
            check_dir(roc_save_epoch_dir)
            roc_save_epoch_cat_dir = os.path.join(roc_save_epoch_dir)
            check_dir(roc_save_epoch_cat_dir)
            mc.save_PTZ_metric2disk(metric_for_conditions[cond_name], roc_save_epoch_cat_dir)
            roc_save_dir = os.path.join(roc_save_epoch_cat_dir,
                                        '_' + str(cond_name) + '_roc.png')
            mc.plotPrecisionRecall(pr, recall, roc_save_dir, benchmark_pr=None)
            f_score_total += f_score

        print(f_score_total / (len(conds)))
        return f_score_total / len(conds)


def main():

    val_transform_det = trans.Compose([
        trans.Scale(256,256),
    ])

    val_data = dates.Dataset('/dataset', '/dataset',
                             '/dataset/test.txt', 'val', transform=True,
                             transform_med=val_transform_det)
    val_loader = Data.DataLoader(val_data, batch_size=1,
                                 shuffle=False, num_workers=4, pin_memory=True)
    import model.siameseNet.dares as models
    model = models.SiameseNet(norm_flag='l2')
    checkpoint = torch.load('the path to best model',
                            map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    print('load success')
    model = model.cuda()
    save_change_map_dir ='the path to changemap'
    save_roc_dir = 'the path to roc'
    time_start = time.time()
    current_metric = validate(model, val_loader, save_change_map_dir, save_roc_dir)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))

if __name__ == '__main__':
   main()

