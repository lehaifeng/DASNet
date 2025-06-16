import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import functional as F
import utils.transforms as trans
import utils.utils as util
import layer.loss as ls
import utils.metric as mc
import shutil
import cv2
from tqdm import tqdm
import cfg.CDD as cfg
import dataset.rs as dates
import time
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 通过 os.environ 获取环境变量

resume = 0

def check_dir(dir):    # 检查是否存在文件夹，没有就创建
    if not os.path.exists(dir):
        os.mkdir(dir)

def untransform(transform_img,mean_vector):

    """
    Pytorch中使用的数据格式与plt.imshow()函数的格式不一致;
    Pytorch中为[Channels, H, W]
    而plt.imshow()中则是[H, W, Channels]
    """
    transform_img = transform_img.transpose(1,2,0) #（channel,h,w）-> (h,w,channel)
    transform_img += mean_vector
    transform_img = transform_img.astype(np.uint8)

    # ::-1表示将字符或数字倒序输出。举个栗子，当line = "abcde"时，使用语句line[::-1]，最后的运行结果为：'edcba'
    transform_img = transform_img[:,:,::-1] # 将通道倒序输出？ 
    return transform_img

def various_distance(out_vec_t0, out_vec_t1,dist_flag): # 一些距离：用于计算变化特征向量映射到某空间的相似度
    """
    F.pairwise_distance、F.cosine_similarity 计算在最后一维的距离
    假设
    输入： x1:(N,C,H,W)、x2:(M,C,H,W)
    输出： 输出:(max(N,M),C,H)
    
    如果要输出每个通道的距离
    应将 X1,X2 view为(N,H,W,C)
    再计算距离
    再view为(N,H,W)
    """
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance

def single_layer_similar_heatmap_visual(output_t0,output_t1,save_change_map_dir,epoch,filename,layer_flag,dist_flag):
    """
    生成某一层的热力图
    热值表示相似度大，冷值表示相似度小
    """
    # interp = nn.functional.interpolate(size=[cfg.TRANSFROM_SCALES[1],cfg.TRANSFROM_SCALES[0]], mode='bilinear')
    n, c, h, w = output_t0.data.shape

    # torch.transpose(Tensor,dim0,dim1)，transpose()一次只能在两个维度间进行转置
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0) # (c,h,w) -> (h*w,c)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0) # (c,h,w) -> (h*w,c)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag) # 在通道维度计算distande(h*w,c) -> (h*w)
    similar_distance_map = distance.view(h,w).data.cpu().numpy() # tensor (h*w) -> ndarry (h,w)
    
    # nn.functional.interpolate 将其插值为给定的size大小
    similar_distance_map_rz = nn.functional.interpolate(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]), # (h,w) -> (1,1,h,w)
                                                        size=[cfg.TRANSFROM_SCALES[1],cfg.TRANSFROM_SCALES[0]], # size: 输出的尺寸大小；scale_factor: 尺寸缩放因子
                                                        mode='bilinear',
                                                        align_corners=True) # 输出可能取决于输入大小，并且不会按照比例将输出和输入像素进行对齐；False为按比例
    

    # cv2.applyColorMap生成伪彩色图像，伪色彩中的 COLORMAP_JET模式，就常被用于生成我们所常见的 热力图
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), # 取(h,w)部分
                                                cv2.COLORMAP_JET)
    save_change_map_dir_ = os.path.join(save_change_map_dir, 'epoch_' + str(epoch))
    check_dir(save_change_map_dir_)
    save_change_map_dir_layer = os.path.join(save_change_map_dir_,layer_flag)
    check_dir(save_change_map_dir_layer)
    save_weight_fig_dir = os.path.join(save_change_map_dir_layer, filename + '.jpg')
    cv2.imwrite(save_weight_fig_dir, similar_dis_map_colorize)
    return similar_distance_map_rz.data.cpu().numpy()

def validate(net, val_dataloader,epoch,save_change_map_dir,save_roc_dir):

    net.eval()
    with torch.no_grad():
        cont_conv5_total,cont_fc_total,cont_embedding_total,num = 0.0,0.0,0.0,0.0
        metric_for_conditions = util.init_metric_for_class_for_cmu(1) # 根据二分类建立一个空的混淆矩阵？m=1代表只有一类为change
        for batch_idx, batch in enumerate(val_dataloader):
            inputs1,input2, targets, filename, height, width = batch
            height, width, filename = height.numpy()[0], width.numpy()[0], filename[0] # tensor([heigth]) -> [heigth] -> height 可换成 int(heigth)
            inputs1,input2,targets = inputs1.cuda(),input2.cuda(), targets.cuda()
            targets = torch.where(targets > 1,torch.ones_like(targets),torch.zeros_like(targets))
            
            # 将数据输入网络得到推理结果
            out_conv5,out_fc,out_embedding = net(inputs1,input2)
            
            # 分配结果
            out_conv5_t0, out_conv5_t1 = out_conv5
            out_fc_t0,out_fc_t1 = out_fc
            out_embedding_t0,out_embedding_t1 = out_embedding
            
            # 计算像素之间的相似度并获得热力图
            conv5_distance_map = single_layer_similar_heatmap_visual(out_conv5_t0,out_conv5_t1,save_change_map_dir,epoch,filename,'conv5','l2')
            fc_distance_map = single_layer_similar_heatmap_visual(out_fc_t0,out_fc_t1,save_change_map_dir,epoch,filename,'fc','l2')
            embedding_distance_map = single_layer_similar_heatmap_visual(out_embedding_t0,out_embedding_t1,save_change_map_dir,epoch,filename,'embedding','l2')
            
            # 计算本批样本的对比度
            cont_conv5 = mc.RMS_Contrast(conv5_distance_map) # RMS_Contrast 计算对比度
            cont_fc = mc.RMS_Contrast(fc_distance_map)
            cont_embedding = mc.RMS_Contrast(embedding_distance_map)
            
            # 计算整体对比度total
            cont_conv5_total += cont_conv5
            cont_fc_total += cont_fc
            cont_embedding_total += cont_embedding
            
            # 计算平均对比度mean
            num += 1
            cont_conv5_mean, cont_fc_mean,cont_embedding_mean = cont_conv5_total/num, \
                                                                                cont_fc_total/num,cont_embedding_total/num

            # 由热力图得到change的概率
            prob_change = embedding_distance_map[0][0] # 取（1,1,h,w）的(h,w)部分
            
            # 获得混淆矩阵
            gt = targets.data.cpu().numpy() 
            FN, FP, posNum, negNum = mc.eval_image_rewrite(gt[0], prob_change, cl_index=1) # 获得本批混淆矩阵的值
            metric_for_conditions[0]['total_fp'] += FP
            metric_for_conditions[0]['total_fn'] += FN
            metric_for_conditions[0]['total_posnum'] += posNum
            metric_for_conditions[0]['total_negnum'] += negNum
            

        thresh = np.array(range(0, 256)) / 255.0
        conds = metric_for_conditions.keys() # 获取字典中所有的keys
        for cond_name in conds:
            
            # 获得val中的混淆矩阵中的值
            total_posnum = metric_for_conditions[cond_name]['total_posnum']
            total_negnum = metric_for_conditions[cond_name]['total_negnum']
            total_fn = metric_for_conditions[cond_name]['total_fn']
            total_fp = metric_for_conditions[cond_name]['total_fp']
            
            # 计算并获得 maxF、precision、recall等度量         
            metric_dict = mc.pxEval_maximizeFMeasure(total_posnum, total_negnum,
                                                    total_fn, total_fp, thresh=thresh)
            
            # 将这些度量存到字典中
            metric_for_conditions[cond_name].setdefault('metric', metric_dict)
            metric_for_conditions[cond_name].setdefault('contrast_conv5', cont_conv5_mean)
            metric_for_conditions[cond_name].setdefault('contrast_fc',cont_fc_mean)
            metric_for_conditions[cond_name].setdefault('contrast_embedding',cont_embedding_mean)

        f_score_total = 0.0
        for cond_name in conds:
            # 获得 precision、recall、MaxF
            pr, recall,f_score = metric_for_conditions[cond_name]['metric']['precision'], metric_for_conditions[cond_name]['metric']['recall'],metric_for_conditions[cond_name]['metric']['MaxF']
            
            roc_save_epoch_dir = os.path.join(save_roc_dir, str(epoch))
            check_dir(roc_save_epoch_dir)
            roc_save_epoch_cat_dir = os.path.join(roc_save_epoch_dir)
            check_dir(roc_save_epoch_cat_dir)
            mc.save_PTZ_metric2disk(metric_for_conditions[cond_name],roc_save_epoch_cat_dir)
            roc_save_dir = os.path.join(roc_save_epoch_cat_dir,
                                            '_' + str(cond_name) + '_roc.png')
            mc.plotPrecisionRecall(pr, recall, roc_save_dir, benchmark_pr=None)
            f_score_total += f_score

        # 获得val的平均 f_score
        print(f_score_total/(len(conds)))
        return f_score_total/len(conds)

def main():

  #########  configs ###########
  best_metric = 0
  ######  load datasets ########
  train_transform_det = trans.Compose([
      trans.Scale(cfg.TRANSFROM_SCALES),
  ])
  val_transform_det = trans.Compose([
      trans.Scale(cfg.TRANSFROM_SCALES),
  ])
  train_data = dates.Dataset(cfg.TRAIN_DATA_PATH,cfg.TRAIN_LABEL_PATH,
                                cfg.TRAIN_TXT_PATH,'train',transform=True,
                                transform_med = train_transform_det)
  train_loader = Data.DataLoader(train_data,batch_size=12
                                 ,
                                 shuffle= True, num_workers= 4, pin_memory= True)
  val_data = dates.Dataset(cfg.VAL_DATA_PATH,cfg.VAL_LABEL_PATH,
                            cfg.VAL_TXT_PATH,'val',transform=True,
                            transform_med = val_transform_det)
  val_loader = Data.DataLoader(val_data, batch_size= cfg.BATCH_SIZE,
                                shuffle= False, num_workers= 4, pin_memory= True)
  ######  build  models ########
  base_seg_model = 'resnet50'
  if base_seg_model == 'vgg':
      import model.siameseNet.d_aa as models
      pretrain_deeplab_path = os.path.join(cfg.PRETRAIN_MODEL_PATH, 'vgg16.pth')
      model = models.SiameseNet(norm_flag='l2')
      if resume:
          checkpoint = torch.load(cfg.TRAINED_BEST_PERFORMANCE_CKPT)
          model.load_state_dict(checkpoint['state_dict'])
          print('resume success')
      else:
          deeplab_pretrain_model = torch.load(pretrain_deeplab_path)
          model.init_parameters_from_deeplab(deeplab_pretrain_model)
          print('load vgg')
  else:
      import model.siameseNet.dares as models
      model = models.SiameseNet(norm_flag='l2')
      if resume:
          checkpoint = torch.load(cfg.TRAINED_BEST_PERFORMANCE_CKPT)
          model.load_state_dict(checkpoint['state_dict'])
          print('resume success')
      else:
          print('load resnet50')

  model = model.cuda()
  MaskLoss = ls.ContrastiveLoss1()
  ab_test_dir = os.path.join(cfg.SAVE_PRED_PATH,'contrastive_loss')
  check_dir(ab_test_dir)
  save_change_map_dir = os.path.join(ab_test_dir, 'changemaps/')
  save_valid_dir = os.path.join(ab_test_dir,'valid_imgs')
  save_roc_dir = os.path.join(ab_test_dir,'roc')
  check_dir(save_change_map_dir),check_dir(save_valid_dir),check_dir(save_roc_dir)
  #########
  ######### optimizer ##########
  ######## how to set different learning rate for differernt layers #########
  optimizer = torch.optim.Adam(params=model.parameters(),lr=cfg.INIT_LEARNING_RATE,weight_decay=cfg.DECAY)
  ######## iter img_label pairs ###########
  loss_total = 0
  time_start = time.time()
  for epoch in range(60):
      for batch_idx, batch in enumerate(train_loader):
            
             step = epoch * len(train_loader) + batch_idx
             util.adjust_learning_rate(cfg.INIT_LEARNING_RATE, optimizer, step)
             model.train()
             img1_idx,img2_idx,label_idx, filename,height,width = batch
             img1,img2,label = img1_idx.cuda(),img2_idx.cuda(),label_idx.cuda()
             label = torch.where(label > 1,torch.ones_like(label),torch.zeros_like(label))
             label = label.float()
             
             
             out_conv5, out_fc,out_embedding = model(img1, img2)

             out_conv5_t0,out_conv5_t1 = out_conv5
             out_fc_t0,out_fc_t1 = out_fc
             out_embedding_t0,out_embedding_t1 = out_embedding
             
             label_rz_conv5 = util.rz_label(label,size=out_conv5_t0.data.cpu().numpy().shape[2:]).cuda()
             label_rz_fc = util.rz_label(label,size=out_fc_t0.data.cpu().numpy().shape[2:]).cuda()
             label_rz_embedding = util.rz_label(label,size=out_embedding_t0.data.cpu().numpy().shape[2:]).cuda()
             
             contractive_loss_conv5 = MaskLoss(out_conv5_t0,out_conv5_t1,label_rz_conv5)
             contractive_loss_fc = MaskLoss(out_fc_t0,out_fc_t1,label_rz_fc)
             contractive_loss_embedding = MaskLoss(out_embedding_t0,out_embedding_t1,label_rz_embedding)
             
             loss = contractive_loss_conv5 + contractive_loss_fc + contractive_loss_embedding
             loss_total += loss.data.cpu()
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
             if (batch_idx) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f Mask_Loss_conv5: %.4f Mask_Loss_fc: %.4f "
                      "Mask_Loss_embedding: %.4f" % (epoch, batch_idx,loss.item(),contractive_loss_conv5.item(),
                                                     contractive_loss_fc.item(),contractive_loss_embedding.item()))
             if (batch_idx) % 1000 == 0:
                 model.eval()
                 current_metric = validate(model, val_loader, epoch,save_change_map_dir,save_roc_dir)
                 if current_metric > best_metric:
                     torch.save({'state_dict': model.state_dict()},
                             os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))
                     shutil.copy(os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'),
                              os.path.join(ab_test_dir, 'model_best.pth'))
                     best_metric = current_metric
      current_metric = validate(model, val_loader, epoch,save_change_map_dir,save_roc_dir)
      if current_metric > best_metric:
         torch.save({'state_dict': model.state_dict()},
                     os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))
         shutil.copy(os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'),
                     os.path.join(ab_test_dir, 'model_best.pth'))
         best_metric = current_metric
      if epoch % 5 == 0:
          torch.save({'state_dict': model.state_dict()},
                       os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))
  elapsed = round(time.time() - time_start)
  elapsed = str(datetime.timedelta(seconds=elapsed))
  print('Elapsed {}'.format(elapsed))

if __name__ == '__main__':
   main()