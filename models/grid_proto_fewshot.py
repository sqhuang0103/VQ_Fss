"""
ALPNet
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpmodule import MultiProtoAsConv, MultiProtoAsConvV2, Allocate
from .sopmodule import SOProtoConv
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder, Resnet_v2
from .backbone.resnet import *
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

# DEBUG
from pdb import set_trace

import pickle
import torchvision

# options for type of prototypes
FG_PROT_MODE = 'gridconv+' # using both local and global prototype
BG_PROT_MODE = 'gridconv'  # using local prototype only. Also 'mask' refers to using global prototype only (as done in vanilla PANet)

# thresholds for deciding class of prototypes
FG_THRESH = 0.55 #0.95
BG_THRESH = 0.95

class FewShotSeg(nn.Module):
    """
    ALPNet
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.get_encoder(in_channels)
        self.get_cls()

    def get_encoder(self, in_channels):
        # if self.config['which_model'] == 'deeplab_res101':
        if self.config['which_model'] == 'dlfcn_res101':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)
        elif self.config['which_model'] == 'resnet_v2':
            pretrain = self.config['use_coco_init']
            self.encoder = resnet101(pretrained=pretrain)
        elif self.config['which_model'] == 'sam':
            device = 'cuda'
            sam = sam_model_registry["vit_h"](checkpoint="/raid/candi/shiqi/sam_pretrained/sam_vit_h_4b8939.pth")
            sam.to(device=device)
            self.encoder = SamPredictor(sam)
        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path))
            print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsConv(proto_grid = [proto_hw, proto_hw], feature_hw =  self.config["feature_hw"]) # when treating it as ordinary prototype
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz = False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            show_viz: return the visualization dictionary
        """
        # ('Please go through this piece of code carefully')
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-shot has not been implemented yet" # NOTE: actual shot in support goes in batch dimension
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]



        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0) #(2,3,256,256)

        img_fts = self.encoder(imgs_concat, low_level = False) #(2,256,32,32)
        fts_size = img_fts.shape[-2:] #(32,32)

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W' #(1,1,1,256,256)
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad = True) #!!! #defalut = True
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        visualizes = [] # the buffer for visualization

        for epi in range(1): # batch dimension, fixed to 1
            fg_masks = [] # keep the way part

            '''
            for way in range(n_ways):
                # note: index of n_ways starts from 0
                mean_sup_ft = supp_fts[way].mean(dim = 0) # [ nb, C, H, W]. Just assume batch size is 1 as pytorch only allows this
                mean_sup_msk = F.interpolate(fore_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear')
                fg_masks.append( mean_sup_msk )

                mean_bg_msk = F.interpolate(back_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear') # [nb, C, H, W]
            '''
            # re-interpolate support mask to the same size as support feature
            res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size = fts_size, mode = 'bilinear',align_corners=True) for fore_mask_w in fore_mask], dim = 0) # [nway, ns, nb, nh', nw']
            res_bg_msk = torch.stack([F.interpolate(back_mask_w, size = fts_size, mode = 'bilinear',align_corners=True) for back_mask_w in back_mask], dim = 0) # [nway, ns, nb, nh', nw']


            scores          = []
            assign_maps     = []
            bg_sim_maps     = []
            fg_sim_maps     = []

            _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode = BG_PROT_MODE, thresh = BG_THRESH, isval = isval, val_wsize = val_wsize, vis_sim = show_viz  )

            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(aux_attr['raw_local_sims'])

            for way, _msk in enumerate(res_fg_msk):
                _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, _msk.unsqueeze(0), mode = FG_PROT_MODE if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', thresh = FG_THRESH, isval = isval, val_wsize = val_wsize, vis_sim = show_viz  )

                scores.append(_raw_score)
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])

            pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear',align_corners=True))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim = 1)
        bg_sim_maps    = torch.stack(bg_sim_maps, dim = 1) if show_viz else None
        fg_sim_maps    = torch.stack(fg_sim_maps, dim = 1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps


    # Batch was at the outer loop
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  #1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        # FIXME: fix this in future we here make a stronger assumption that a positive class must be there to avoid undersegmentation/ lazyness
        skip_ways = []

        ### added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2) # added to nway(1) and nb(1)

        ### end of added part

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1] # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]

                qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size = img_fts.shape[-2:], mode = 'bilinear',align_corners=True) # [1 (way), n (shot), h, w]

                # background
                qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size = img_fts.shape[-2:], mode = 'bilinear',align_corners=True) # 1, n, h ,w
                scores = []

                _raw_score_bg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_bg_msk.unsqueeze(-3), mode = BG_PROT_MODE, thresh = BG_THRESH )

                scores.append(_raw_score_bg)

                _raw_score_fg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_fg_msk.unsqueeze(-3), mode = FG_PROT_MODE if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', thresh = FG_THRESH )
                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear',align_corners=True)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss.append( F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)

        return torch.sum( torch.stack(loss))

class FewShotSegV2(FewShotSeg):
    def __init__(self,in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSegV2, self).__init__(in_channels, pretrained_path, cfg)

    def get_encoder(self, in_channels):
        # if self.config['which_model'] == 'deeplab_res101':
        if self.config['which_model'] == 'dlfcn_res101':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)
        elif self.config['which_model'] == 'resnet_v2':
            pretrain = self.config['use_coco_init']
            self.encoder = Resnet_v2(pretrain)
        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path))
            print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsConvV2(proto_grid = [proto_hw, proto_hw], feature_hw = self.config["feature_hw"]) # when treating it as ordinary prototype
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')

    # Batch was at the outer loop
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  #1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        skip_ways = []

        ### added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2) # added to nway(1) and nb(1)

        ### end of added part

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1] # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]

                qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size = img_fts.shape[-2:], mode = 'bilinear',align_corners=True) # [1 (way), n (shot), h, w]

                # background
                qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size = img_fts.shape[-2:], mode = 'bilinear',align_corners=True) # 1, n, h ,w
                scores = []

                # _raw_score_bg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_bg_msk.unsqueeze(-3), mode = BG_PROT_MODE, thresh = BG_THRESH )

                # scores.append(_raw_score_bg)

                # _raw_score_fg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_fg_msk.unsqueeze(-3), mode = FG_PROT_MODE if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', thresh = FG_THRESH )
                # scores.append(_raw_score_fg)

                BG_protos = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_bg_msk.unsqueeze(-3), mode = BG_PROT_MODE, thresh = BG_THRESH)
                _bg_raw_score, _, bg_aux_attr = self.cls_unit.CalcDist(BG_PROT_MODE, BG_protos, vis_sim=False)

                # for way, _msk in enumerate(res_fg_msk):
                if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask':
                    FG_protos = self.cls_unit(img_fts, qry_fts, qry_pred_fg_msk.unsqueeze(-3),
                                              mode=FG_PROT_MODE,
                                              thresh=FG_THRESH)
                    # lvq_BG_protos, lvq_FG_protos = self.cls_unit.LVQ([BG_protos, FG_protos], 8)
                    # BG_protos = torch.cat([BG_protos,lvq_BG_protos],0)
                    # FG_protos = torch.cat([FG_protos,lvq_FG_protos],0)
                    _fg_raw_score, _, fg_aux_attr = self.cls_unit.CalcDist(FG_PROT_MODE, FG_protos, vis_sim=False)


                else:
                    FG_protos = self.cls_unit(img_fts, qry_fts, qry_pred_fg_msk.unsqueeze(-3),
                                              mode='mask',
                                              thresh=FG_THRESH)
                    _fg_raw_score, _, fg_aux_attr = self.cls_unit.CalcDist('mask', FG_protos, vis_sim=False)
                scores.append(_bg_raw_score)
                scores.append(_fg_raw_score)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear',align_corners=True)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss.append( F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)

        return torch.sum( torch.stack(loss))



    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            show_viz: return the visualization dictionary
        """
        # ('Please go through this piece of code carefully')
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-shot has not been implemented yet"  # NOTE: actual shot in support goes in batch dimension
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)  # (2,3,256,256)
        img_fts = self.encoder(imgs_concat, low_level=False)  # (2,256,32,32)
        fts_size = img_fts.shape[-2:]  # (32,32)

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)  # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W' #(1,1,1,256,256)
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad=True)  # !!! #defalut = True
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        visualizes = []  # the buffer for visualization

        for epi in range(1):  # batch dimension, fixed to 1
            fg_masks = []  # keep the way part

            '''
            for way in range(n_ways):
                # note: index of n_ways starts from 0
                mean_sup_ft = supp_fts[way].mean(dim = 0) # [ nb, C, H, W]. Just assume batch size is 1 as pytorch only allows this
                mean_sup_msk = F.interpolate(fore_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear')
                fg_masks.append( mean_sup_msk )

                mean_bg_msk = F.interpolate(back_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear') # [nb, C, H, W]
            '''
            # re-interpolate support mask to the same size as support feature
            res_fg_msk = torch.stack(
                [F.interpolate(fore_mask_w, size=fts_size, mode='bilinear',align_corners=True) for fore_mask_w in fore_mask],
                dim=0)  # [nway, ns, nb, nh', nw']
            res_bg_msk = torch.stack(
                [F.interpolate(back_mask_w, size=fts_size, mode='bilinear',align_corners=True) for back_mask_w in back_mask],
                dim=0)  # [nway, ns, nb, nh', nw']

            scores = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []
            som_map = None

            # vis_protos = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode='gridall', thresh=BG_THRESH,
            #                                         isval=isval, val_wsize=val_wsize, vis_sim=show_viz)


            # _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode=BG_PROT_MODE, thresh=BG_THRESH,
            #                                         isval=isval, val_wsize=val_wsize, vis_sim=show_viz)
            ################# BG_proto #############################################
            BG_protos = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode=BG_PROT_MODE, thresh=BG_THRESH,
                                                    isval=isval, val_wsize=val_wsize, vis_sim=show_viz)

            _bg_raw_score, _, bg_aux_attr = self.cls_unit.CalcDist(BG_PROT_MODE, BG_protos, vis_sim=False)


            ################## FG proto ###################################################
            if self.config['lvq_function'] == 'feat_for_proto':
                lvq_function = self.cls_unit.LVQ_feat_for_proto
            elif self.config['lvq_function'] == 'proto_for_feat':
                lvq_function = self.cls_unit.LVQ_proto_for_feat
            elif self.config['lvq_function'] == 'feat_for_feat':
                lvq_function = self.cls_unit.LVQ_feat_for_feat
            else:
                lvq_function = self.cls_unit.LVQ_feat_for_proto

            for way, _msk in enumerate(res_fg_msk):
                if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask':
                    FG_protos = self.cls_unit(qry_fts, supp_fts, _msk.unsqueeze(0),
                                                        mode=FG_PROT_MODE,
                                                        thresh=FG_THRESH, isval=isval, val_wsize=val_wsize,
                                                        vis_sim=show_viz)
                    if self.config['sommode'] =='som':
                        som_BG_protos, som_FG_protos, som_map = self.cls_unit.SOM_labeledV3(nerous=self.config['som_nerous'],iter=self.config['som_iter'],exclude=False, BGleft=False)
                        if self.config['weight_func']:
                            mapping = Allocate(self.config['weight_func'])
                            # som_BG_protos = mapping.cal(som_BG_protos, BG_protos)
                            som_FG_protos = mapping.cal(som_FG_protos, torch.vstack((FG_protos,BG_protos)) )
                        if self.config['lvqmode'] == 'som':
                            lvq_BG_protos, lvq_FG_protos = self.cls_unit.LVQ_feat_for_proto([som_BG_protos, som_FG_protos], num_proto=self.config['lvq_num_proto'],max_iter=self.config['lvq_maxiter'])
                            BG_protos = torch.cat([BG_protos, som_BG_protos, lvq_BG_protos], 0)
                            FG_protos = torch.cat([FG_protos, som_FG_protos, lvq_FG_protos], 0)
                        elif self.config['lvqmode'] == 'grid':
                            lvq_BG_protos, lvq_FG_protos = self.cls_unit.LVQ_feat_for_proto([BG_protos, FG_protos],
                                                                                            num_proto=self.config['lvq_num_proto'],max_iter=self.config['lvq_maxiter'])
                            BG_protos = torch.cat([BG_protos, som_BG_protos, lvq_BG_protos], 0)
                            FG_protos = torch.cat([FG_protos, som_FG_protos, lvq_FG_protos], 0)
                        elif self.config['lvqmode'] == 'both':
                            BG_protos = torch.cat([BG_protos, som_BG_protos], 0)
                            FG_protos = torch.cat([FG_protos, som_FG_protos], 0)
                            lvq_BG_protos, lvq_FG_protos = lvq_function([BG_protos, FG_protos], num_proto=self.config['lvq_num_proto'],max_iter=self.config['lvq_maxiter'])
                            BG_protos = torch.cat([BG_protos, lvq_BG_protos], 0)
                            FG_protos = torch.cat([FG_protos, lvq_FG_protos], 0)
                        else:
                            # BG_protos = torch.cat([BG_protos, som_BG_protos], 0)
                            FG_protos = torch.cat([FG_protos, som_FG_protos], 0)

                    # lvq_BG_protos, lvq_FG_protos = self.cls_unit.LVQ([BG_protos, FG_protos], num_proto=8)
                    # som_BG_protos, som_FG_protos = self.cls_unit.SOM_labeled(nerous=7)
                    # som_BG_protos, som_FG_protos = self.cls_unit.SOM_labeledV2(nerous=7,exclude=False, BGleft=False)
                    # lvq_BG_protos, lvq_FG_protos = self.cls_unit.LVQ_feat_for_proto([BG_protos, FG_protos], 1)
                    # lvq_BG_protos, lvq_FG_protos = self.cls_unit.LVQ_feat_for_feat(num_proto=2)
                    #================================================================================================
                    # BG_protos = torch.cat([BG_protos,som_BG_protos],0)
                    # FG_protos = torch.cat([FG_protos,som_FG_protos],0)
                    # lvq_BG_protos, lvq_FG_protos = self.cls_unit.LVQ_feat_for_feat([BG_protos, FG_protos], 2)
                    # BG_protos = torch.cat([BG_protos, lvq_BG_protos], 0)
                    # FG_protos = torch.cat([FG_protos, lvq_FG_protos], 0)
                    # ==============================================================================================
                    _fg_raw_score, _, fg_aux_attr = self.cls_unit.CalcDist(FG_PROT_MODE, FG_protos, vis_sim=False)
                else:
                    # FG_protos = self.cls_unit(qry_fts, supp_fts, _msk.unsqueeze(0),
                    #                           mode=FG_PROT_MODE,
                    #                           thresh=FG_THRESH, isval=isval, val_wsize=val_wsize,
                    #                           vis_sim=show_viz)
                    FG_protos = self.cls_unit(qry_fts, supp_fts, _msk.unsqueeze(0),
                                                        mode='mask',
                                                        thresh=FG_THRESH, isval=isval, val_wsize=val_wsize,
                                                        vis_sim=show_viz)

                    _fg_raw_score, _, fg_aux_attr = self.cls_unit.CalcDist(FG_PROT_MODE, FG_protos, vis_sim=False)
                    # _fg_raw_score, _, fg_aux_attr = self.cls_unit.CalcDist('mask',FG_protos, vis_sim=False)
            if self.config['bgmode'] == 'preprocessed':
                _bg_raw_score, _, bg_aux_attr = self.cls_unit.CalcDist(BG_PROT_MODE, BG_protos, vis_sim=False)

            scores.append(_bg_raw_score)
            assign_maps.append(fg_aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(bg_aux_attr['raw_local_sims'])
            scores.append(_fg_raw_score)
            if show_viz:
                fg_sim_maps.append(fg_aux_attr['raw_local_sims'])


            pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear',align_corners=True))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim=1)
        # if len(som_maps) > 0:
        #     som_maps = torch.stack(som_maps, dim=1)
        # else:
        #     som_maps = False
        bg_sim_maps = torch.stack(bg_sim_maps, dim=1) if show_viz else None
        fg_sim_maps = torch.stack(fg_sim_maps, dim=1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps #, r #, som_map

class FewShotSegV3(FewShotSegV2):
    def __init__(self,in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSegV3, self).__init__(in_channels, pretrained_path, cfg)

    def get_encoder(self, in_channels):
        # if self.config['which_model'] == 'deeplab_res101':
        if self.config['which_model'] == 'dlfcn_res101':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)
        elif self.config['which_model'] == 'resnet_v2':
            pretrain = self.config['use_coco_init']
            self.encoder = Resnet_v2(pretrain)
        elif self.config['which_model'] == 'sam':
            device = 'cuda'
            sam = sam_model_registry["vit_h"](checkpoint="/raid/candi/shiqi/sam_pretrained/sam_vit_h_4b8939.pth")
            sam.to(device=device)
            self.encoder = SamPredictor(sam)
            self.sam = True
            self.pretrained_path = False
        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path))
            print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            # self.cls_unit = MultiProtoAsConvV2(proto_grid = [proto_hw, proto_hw], feature_hw = self.config["feature_hw"]) # when treating it as ordinary prototype
            self.cls_unit = SOProtoConv(proto_grid = [proto_hw, proto_hw], feature_hw = self.config["feature_hw"]) # when treating it as ordinary prototype
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')

    def sam_encoder(self,img_concat):
        # img_concat (2,3,256,256)
        sup_img = img_concat[0]
        qry_img = img_concat[1]
        sup_img = sup_img.permute(1,2,0)
        qry_img = qry_img.permute(1,2,0)
        sup_img = sup_img.detach().cpu().numpy()
        qry_img = qry_img.detach().cpu().numpy()
        sup_img = np.uint8(sup_img)
        qry_img = np.uint8(qry_img)
        assert sup_img.shape[2] == 3
        self.encoder.set_image(sup_img)
        sup_fts = self.encoder.get_image_embedding()# [1, 256, 64, 64]
        self.encoder.set_image(qry_img)
        qry_fts = self.encoder.get_image_embedding() # [1, 256, 64, 64]
        return torch.cat((sup_fts,qry_fts),dim=0)




    def selfVQ(self, B_protos, F_protos):
        som_BG_protos, som_FG_protos, som_map = self.cls_unit.SOM_labeledV2(nerous=self.config['som_nerous'],
                                                                            iter=self.config['som_iter'], exclude=False,
                                                                            BGleft=False)
        # import pdb
        # pdb.set_trace()
        F_protos = torch.cat([F_protos, som_FG_protos], 0)
        # B_protos = torch.cat([B_protos, som_BG_protos], 0)
        return B_protos, F_protos

    def ResVQ(self,lvq_function, B_protos, F_protos):
        lvq_BG_protos, lvq_FG_protos = lvq_function([B_protos, F_protos], num_proto=self.config['lvq_num_proto'],
                                                    max_iter=self.config['lvq_maxiter'])
        F_protos = torch.cat([F_protos, lvq_FG_protos], 0)
        B_protos = torch.cat([B_protos, lvq_BG_protos], 0)
        return B_protos, F_protos

    # Batch was at the outer loop
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  #1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        skip_ways = []

        ### added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2) # added to nway(1) and nb(1)

        ### end of added part

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1] # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]

                qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size = img_fts.shape[-2:], mode = 'bilinear',align_corners=True) # [1 (way), n (shot), h, w]

                # background
                qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size = img_fts.shape[-2:], mode = 'bilinear',align_corners=True) # 1, n, h ,w
                scores = []

                # _raw_score_bg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_bg_msk.unsqueeze(-3), mode = BG_PROT_MODE, thresh = BG_THRESH )

                # scores.append(_raw_score_bg)

                # _raw_score_fg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_fg_msk.unsqueeze(-3), mode = FG_PROT_MODE if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', thresh = FG_THRESH )
                # scores.append(_raw_score_fg)

                BG_protos = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_bg_msk.unsqueeze(-3), mode = BG_PROT_MODE, thresh = BG_THRESH)
                _bg_raw_score, _, bg_aux_attr = self.cls_unit.CalcDist(BG_PROT_MODE, BG_protos, vis_sim=False)

                # for way, _msk in enumerate(res_fg_msk):
                if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask':
                    FG_protos = self.cls_unit(img_fts, qry_fts, qry_pred_fg_msk.unsqueeze(-3),
                                              mode=FG_PROT_MODE,
                                              thresh=FG_THRESH)
                    # lvq_BG_protos, lvq_FG_protos = self.cls_unit.LVQ([BG_protos, FG_protos], 8)
                    # BG_protos = torch.cat([BG_protos,lvq_BG_protos],0)
                    # FG_protos = torch.cat([FG_protos,lvq_FG_protos],0)
                    _fg_raw_score, _, fg_aux_attr = self.cls_unit.CalcDist(FG_PROT_MODE, FG_protos, vis_sim=False)


                else:
                    FG_protos = self.cls_unit(img_fts, qry_fts, qry_pred_fg_msk.unsqueeze(-3),
                                              mode='mask',
                                              thresh=FG_THRESH)
                    _fg_raw_score, _, fg_aux_attr = self.cls_unit.CalcDist('mask', FG_protos, vis_sim=False)
                scores.append(_bg_raw_score)
                scores.append(_fg_raw_score)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear',align_corners=True)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss.append( F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)

        return torch.sum( torch.stack(loss))

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz=False):
        """
                Args:
                    supp_imgs: support images
                        way x shot x [B x 3 x H x W], list of lists of tensors
                    fore_mask: foreground masks for support images
                        way x shot x [B x H x W], list of lists of tensors
                    back_mask: background masks for support images
                        way x shot x [B x H x W], list of lists of tensors
                    qry_imgs: query images
                        N x [B x 3 x H x W], list of tensors
                    show_viz: return the visualization dictionary
                """
        # ('Please go through this piece of code carefully')
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-shot has not been implemented yet"  # NOTE: actual shot in support goes in batch dimension
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)  # (2,3,256,256)
        if self.config['which_model'] == 'sam':
            img_fts = self.sam_encoder(imgs_concat)
        else:
            img_fts = self.encoder(imgs_concat, low_level=False)  # (2,256,32,32)
        fts_size = img_fts.shape[-2:]  # (32,32)

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)  # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W' #(1,1,1,256,256)
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad=True)  # !!! #defalut = True
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        visualizes = []  # the buffer for visualization

        for epi in range(1):  # batch dimension, fixed to 1
            fg_masks = []  # keep the way part

            # re-interpolate support mask to the same size as support feature
            res_fg_msk = torch.stack(
                [F.interpolate(fore_mask_w, size=fts_size, mode='bilinear', align_corners=True) for fore_mask_w in
                 fore_mask],
                dim=0)  # [nway, ns, nb, nh', nw']
            res_bg_msk = torch.stack(
                [F.interpolate(back_mask_w, size=fts_size, mode='bilinear', align_corners=True) for back_mask_w in
                 back_mask],
                dim=0)  # [nway, ns, nb, nh', nw']

            scores = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []
            som_map = None

            # vis_protos = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode='gridall', thresh=BG_THRESH,
            #                                         isval=isval, val_wsize=val_wsize, vis_sim=show_viz)

            # _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode=BG_PROT_MODE, thresh=BG_THRESH,
            #                                         isval=isval, val_wsize=val_wsize, vis_sim=show_viz)
            ################# BG_proto #############################################
            BG_protos = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode=BG_PROT_MODE, thresh=BG_THRESH,
                                      isval=isval, val_wsize=val_wsize, vis_sim=show_viz)

            _bg_raw_score, _, bg_aux_attr = self.cls_unit.CalcDist(BG_PROT_MODE, BG_protos, vis_sim=False)

            ################## FG proto ###################################################
            if self.config['lvq_function'] == 'feat_for_proto':
                lvq_function = self.cls_unit.LVQ_feat_for_proto
            elif self.config['lvq_function'] == 'proto_for_feat':
                lvq_function = self.cls_unit.LVQ_proto_for_feat
            elif self.config['lvq_function'] == 'feat_for_feat':
                lvq_function = self.cls_unit.LVQ_feat_for_feat
            else:
                lvq_function = self.cls_unit.LVQ_feat_for_feat

            for way, _msk in enumerate(res_fg_msk):
                if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask':
                    FG_protos = self.cls_unit(qry_fts, supp_fts, _msk.unsqueeze(0),
                                              mode=FG_PROT_MODE,
                                              thresh=FG_THRESH, isval=isval, val_wsize=val_wsize,
                                              vis_sim=show_viz)
                    if self.config['sommode'] == 'som':
                        BG_protos, s_FG_protos = self.selfVQ(BG_protos,FG_protos)
                    _fg_raw_score, _, fg_aux_attr = self.cls_unit.CalcDist(FG_PROT_MODE, FG_protos, vis_sim=False)

                else:
                    FG_protos = self.cls_unit(qry_fts, supp_fts, _msk.unsqueeze(0),
                                                        mode='mask',
                                                        thresh=FG_THRESH, isval=isval, val_wsize=val_wsize,
                                                        vis_sim=show_viz)

                    _fg_raw_score, _, fg_aux_attr = self.cls_unit.CalcDist(FG_PROT_MODE, FG_protos, vis_sim=False)

            if self.config['bgmode'] == 'preprocessed':
                _bg_raw_score, _, bg_aux_attr = self.cls_unit.CalcDist(BG_PROT_MODE, BG_protos, vis_sim=False)

            scores.append(_bg_raw_score)
            assign_maps.append(fg_aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(bg_aux_attr['raw_local_sims'])
            scores.append(_fg_raw_score)
            if show_viz:
                fg_sim_maps.append(fg_aux_attr['raw_local_sims'])

            pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim=1)
        # if len(som_maps) > 0:
        #     som_maps = torch.stack(som_maps, dim=1)
        # else:
        #     som_maps = False
        bg_sim_maps = torch.stack(bg_sim_maps, dim=1) if show_viz else None
        fg_sim_maps = torch.stack(fg_sim_maps, dim=1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps  # , r #, som_map










