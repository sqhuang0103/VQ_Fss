"""
ALPNet
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .alpmodule import MultiProtoAsConv, MultiProtoAsConvV2, Allocate
from .sopmodule import SOProtoConv
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder, Resnet_v2
from .backbone.resnet import *
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from minisom import MiniSom
from sklearn_lvq import GmlvqModel
from sklearn.cluster import KMeans

# DEBUG
from pdb import set_trace

import pickle
import torchvision

# options for type of prototypes
FG_PROT_MODE = 'kmeans' #'som+' #'gridconv+' # using both local and global prototype
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


            scores = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []

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
        bg_sim_maps = torch.stack(bg_sim_maps, dim = 1) if show_viz else None
        fg_sim_maps = torch.stack(fg_sim_maps, dim = 1) if show_viz else None

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

class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode = 'bilinear'):
        """
        ALPModule
        Args:
            proto_grid:     Grid size when doing multi-prototyping. For a 32-by-32 feature map, a size of 16-by-16 leads to a pooling window of 2-by-2
            feature_hw:     Spatial size of input feature map

        """
        super(MultiProtoAsConv, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        kernel_size = [ ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)  ]
        self.avg_pool_op = nn.AvgPool2d( kernel_size  )

    def forward(self, qry, sup_x, sup_y, mode, thresh, isval = False, val_wsize = None, vis_sim = False, **kwargs):
        """
        Now supports
        Args:
            mode: 'mask'/ 'grid'. if mask, works as original prototyping
            qry: [way(1), nc, h, w]
            sup_x: [nb, nc, h, w]
            sup_y: [nb, 1, h, w]
            vis_sim: visualize raw similarities or not
        New
            mode:       'mask'/ 'grid'. if mask, works as original prototyping
            qry:        [way(1), nb(1), nc, h, w]
            sup_x:      [way(1), shot, nb(1), nc, h, w]
            sup_y:      [way(1), shot, nb(1), h, w]
            vis_sim:    visualize raw similarities or not
        """

        qry = qry.squeeze(1) # [way(1), nb(1), nc, hw] -> [way(1), nc, h, w]
        sup_x = sup_x.squeeze(0).squeeze(1) # [nshot, nc, h, w]
        sup_y = sup_y.squeeze(0) # [nshot, 1, h, w]

        def safe_norm(x, p = 2, dim = 1, eps = 1e-4):
            x_norm = torch.norm(x, p = p, dim = dim) # .detach()
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        if mode == 'mask': # class-level prototype only
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5) # nb x C

            proto = proto.mean(dim = 0, keepdim = True) # 1 X C, take the mean of everything
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps = 1e-4) * 20.0 # [1, h, w]

            vis_dict = {'proto_assign': None} # things to visualize
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict  # just a placeholder. pred_mask returned as [1, way(1), h, w]

        # no need to merge with gridconv+
        elif mode == 'gridconv': # using local prototypes only

            input_size = qry.shape
            nch = input_size[1]

            sup_nshot = sup_x.shape[0] #(1,256,32,32)

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  ) #(1,256,8.8)

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0) # way(1),nb, hw, nc
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)


            protos = n_sup_x[sup_y_g > thresh, :] # npro, nc #thresh 0.95
            pro_n = safe_norm(protos)
            qry_n = safe_norm(qry)

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            debug_assign = dists.argmax(dim = 1).float().detach()

            vis_dict = {'proto_assign': debug_assign} # things to visualize

            if vis_sim: # return the similarity for visualization
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict


        elif mode == 'gridconv+': # local and global prototypes

            input_size = qry.shape # torch.Size([1, 256, 32, 32])
            nch = input_size[1]
            nb_q = input_size[0]

            sup_size = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  ) # torch.Size([1, 256, 8, 8])

            sup_nshot = sup_x.shape[0]

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0) # torch.Size([1, 1, 64, 256])
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0) # torch.Size([1, 1, 64, 256])

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)

            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0) # torch.Size([1, 1, 64])

            protos = n_sup_x[sup_y_g > thresh, :] # torch.Size([1, 256]) it can be (n_proto,256)


            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5) # torch.Size([1, 256])

            pro_n = safe_norm( torch.cat( [protos, glb_proto], dim = 0 ) ) # torch.Size([2, 256])

            qry_n = safe_norm(qry) # torch.Size([1, 256, 32, 32])

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20 # torch.Size([1, 2, 32, 32])

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True) # torch.Size([1, 1, 32, 32])
            raw_local_sims = dists.detach()


            debug_assign = dists.argmax(dim = 1).float() # torch.Size([1, 32, 32])

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict

        elif mode == 'kmeans': # local and global prototypes

            input_size = qry.shape # torch.Size([1, 256, 32, 32])
            nch = input_size[1]


            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  ) # torch.Size([1, 256, 8, 8])

            sup_nshot = sup_x.shape[0]

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0) # torch.Size([1, 1, 64, 256])
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0) # torch.Size([1, 1, 64, 256])

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)

            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0) # torch.Size([1, 1, 64])

            #kmeans implementation
            protos = lvq_fg(sup_x,3,sup_y)

            # protos = n_sup_x[sup_y_g > thresh, :] # torch.Size([1, 256])


            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5) # torch.Size([1, 256])

            pro_n = safe_norm( torch.cat( [protos, glb_proto], dim = 0 ) ) # torch.Size([2, 256])

            qry_n = safe_norm(qry) # torch.Size([1, 256, 32, 32])

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20 # torch.Size([1, 2, 32, 32])

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True) # torch.Size([1, 1, 32, 32])
            raw_local_sims = dists.detach()


            debug_assign = dists.argmax(dim = 1).float() # torch.Size([1, 32, 32])

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict

        else:
            raise NotImplementedError


def kmeans_fg(input_vectors, num_prototypes, label_mask):
    """
    Generate prototypes from input vectors using K-means clustering based on the label mask.

    Parameters:
        input_vectors: torch tensor of shape (1, 256, 32, 32) representing input vectors.
        num_prototypes: Total number of prototypes to generate.
        label_mask: torch tensor of shape (1, 1, 32, 32) representing the binary label mask.

    Returns:
        prototypes: torch tensor of shape (1, 256, num_prototypes) representing generated prototypes.
    """
    # Reshape input vectors to (N, D) where N is the total number of vectors and D is the dimensionality
    N, D = input_vectors.shape[2]*input_vectors.shape[3], input_vectors.shape[1]
    flattened_vectors = input_vectors.view(N, D)

    # Get the indices of foreground vectors based on the label mask
    foreground_indices = torch.nonzero(label_mask).squeeze()

    # Select only the foreground vectors
    foreground_vectors = flattened_vectors[foreground_indices]

    # Convert to numpy array and move to CPU
    foreground_vectors_np = foreground_vectors.detach().cpu().numpy()

    # Perform K-means clustering to generate prototypes
    kmeans = KMeans(n_clusters=num_prototypes, random_state=0)
    # print(foreground_vectors_np.shape) # (15, 4, 256)
    foreground_vectors_np = foreground_vectors_np.reshape(-1, D)
    kmeans.fit(foreground_vectors_np)
    cluster_centers = kmeans.cluster_centers_

    # Convert cluster centers back to torch tensor and move to the same device as input_vectors
    prototypes = torch.tensor(cluster_centers, dtype=input_vectors.dtype, device=input_vectors.device)
    prototypes = prototypes.view(num_prototypes, D )

    return prototypes

def som_fg(input_vectors, num_prototypes, label_mask):
    """
        Generate prototypes from input vectors using K-means clustering based on the label mask.

        Parameters:
            input_vectors: torch tensor of shape (1, 256, 32, 32) representing input vectors.
            num_prototypes: Total number of prototypes to generate.
            label_mask: torch tensor of shape (1, 1, 32, 32) representing the binary label mask.

        Returns:
            prototypes: torch tensor of shape (1, 256, num_prototypes) representing generated prototypes.
        """
    # Reshape input vectors to (N, D) where N is the total number of vectors and D is the dimensionality
    N, D = input_vectors.shape[2] * input_vectors.shape[3], input_vectors.shape[1]
    flattened_vectors = input_vectors.view(N, D)

    # Get the indices of foreground vectors based on the label mask
    foreground_indices = torch.nonzero(label_mask).squeeze()

    # Select only the foreground vectors
    foreground_vectors = flattened_vectors[foreground_indices]

    # Convert to numpy array and move to CPU
    foreground_vectors_np = foreground_vectors.detach().cpu().numpy()
    foreground_vectors_np = foreground_vectors_np.reshape(-1, D)

    # Define the grid size for the SOM
    grid_size = (3, 3)  # Example grid size, adjust as needed
    # Initialize the SOM
    som = MiniSom(grid_size[0], grid_size[1], 256, sigma=0.5, learning_rate=0.5)  # Adjust parameters as needed
    # Train the SOM
    som.train_random(foreground_vectors_np, 100)  # Adjust number of iterations as needed
    # Get the prototype vectors from the SOM
    prototypes = som.get_weights()

    # Convert cluster centers back to torch tensor and move to the same device as input_vectors
    prototypes = torch.tensor(prototypes, dtype=input_vectors.dtype, device=input_vectors.device)
    prototypes = prototypes.view(grid_size[0]*grid_size[1], D)

    return prototypes

def lvq_fg(input_vectors, num_prototypes, label_mask):
    """
    Generate prototypes to distinguish between foreground and background using Learning Vector Quantization (LVQ).

    Parameters:
        input_vectors: torch tensor of shape (1, 256, 32, 32) representing input vectors.
        num_prototypes: Total number of prototypes to generate.
        label_mask: torch tensor of shape (1, 1, 32, 32) representing the binary label mask.

    Returns:
        prototypes: torch tensor of shape (1, 256, num_prototypes) representing the generated prototypes.
    """
    # Reshape input vectors to (N, D) where N is the total number of vectors and D is the dimensionality
    N, D = input_vectors.shape[2]*input_vectors.shape[3], input_vectors.shape[1]
    flattened_vectors = input_vectors.view(N, D)

    # Get the indices of foreground and background vectors based on the label mask
    foreground_indices = torch.nonzero(label_mask).squeeze()
    background_indices = torch.nonzero(1 - label_mask).squeeze()

    # Select foreground and background vectors
    foreground_vectors = flattened_vectors[foreground_indices]
    background_vectors = flattened_vectors[background_indices]
    foreground_vectors = foreground_vectors.detach().cpu().numpy()
    background_vectors = background_vectors.detach().cpu().numpy()
    foreground_vectors = foreground_vectors.reshape(-1, D)
    background_vectors = background_vectors.reshape(-1, D)

    # Create a dataset with foreground and background vectors and their labels (0 for background, 1 for foreground)
    dataset = [(vector, 1) for vector in foreground_vectors] + [(vector, 0) for vector in background_vectors]

    # Convert to numpy array and move to CPU
    X = np.array([data[0] for data in dataset])
    y = np.array([data[1] for data in dataset])

    # Perform LVQ to generate prototypes
    lvq = GmlvqModel()
    lvq.fit(X, y)
    fg_prototypes = lvq.w_[1,:]
    bg_prototypes = lvq.w_[0,:]
    fg_prototypes = fg_prototypes[None,...]
    prototypes = torch.tensor(fg_prototypes, dtype=input_vectors.dtype, device=input_vectors.device)

    return prototypes






