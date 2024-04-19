"""
ALPModule
"""
import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
# for unit test from spatial_similarity_module import NONLocalBlock2D, LayerNorm
from sklearn_lvq import RslvqModel,GlvqModel
import math

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

            input_size = qry.shape
            nch = input_size[1]
            nb_q = input_size[0]

            sup_size = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  )

            sup_nshot = sup_x.shape[0]

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)

            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]


            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5)

            pro_n = safe_norm( torch.cat( [protos, glb_proto], dim = 0 ) )

            qry_n = safe_norm(qry)

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            raw_local_sims = dists.detach()


            debug_assign = dists.argmax(dim = 1).float()

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict

        else:
            raise NotImplementedError


class MultiProtoAsConvV2(MultiProtoAsConv):
    def __init__(self,proto_grid, feature_hw, upsample_mode = 'bilinear'):
        super(MultiProtoAsConvV2, self).__init__(proto_grid, feature_hw, upsample_mode)

    def forward(self, qry, sup_x, sup_y, mode, thresh, isval = False, val_wsize = None, vis_sim = False, **kwargs):
        self.qry = qry.squeeze(1)  # [way(1), nb(1), nc, hw] -> [way(1), nc, h, w]
        self.sup_x = sup_x.squeeze(0).squeeze(1)  # [nshot, nc, h, w]
        self.sup_y = sup_y.squeeze(0)  # [nshot, 1, h, w]

        def safe_norm(x, p=2, dim=1, eps=1e-4):
            x_norm = torch.norm(x, p=p, dim=dim)  # .detach()
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        if mode == 'mask':  # class-level prototype only

            proto = torch.sum(self.sup_x * self.sup_y, dim=(-1, -2)) \
                    / (self.sup_y.sum(dim=(-1, -2)) + 1e-5)  # nb x C

            proto = proto.mean(dim=0, keepdim=True)  # 1 X C, take the mean of everything
            # pred_mask = F.cosine_similarity(self.qry, proto[..., None, None], dim=1, eps=1e-4) * 20.0  # [1, h, w]

            # vis_dict = {'proto_assign': None}  # things to visualize
            # if vis_sim:
            #     vis_dict['raw_local_sims'] = pred_mask
            # return pred_mask.unsqueeze(1), [
            #     pred_mask], vis_dict  # just a placeholder. pred_mask returned as [1, way(1), h, w]
            return proto
        # no need to merge with gridconv+
        elif mode == 'gridconv':  # using local prototypes only

            input_size = self.qry.shape
            nch = input_size[1]

            sup_nshot = self.sup_x.shape[0]  # (1,256,32,32)

            n_sup_x = F.avg_pool2d(self.sup_x, val_wsize) if isval else self.avg_pool_op(self.sup_x)  # (1,256,8.8)

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)  # way(1),nb, hw, nc
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(self.sup_y, val_wsize) if isval else self.avg_pool_op(self.sup_y)
            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]  # npro, nc #thresh 0.95

            return protos

        elif mode == 'gridall':
            input_size = self.qry.shape
            nch = input_size[1]

            sup_nshot = self.sup_x.shape[0]  # (1,256,32,32)
            n_sup_x = self.avg_pool_op(self.qry)  # (1,256,8,8)
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)  # way(1),nb, hw, nc
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            return n_sup_x


        elif mode == 'gridconv+':  # local and global prototypes

            input_size = self.qry.shape
            nch = input_size[1]
            nb_q = input_size[0]


            sup_size = self.sup_x.shape[0]

            # n_sup_x = F.avg_pool2d(self.sup_x, val_wsize) if isval else self.avg_pool_op(self.sup_x)
            n_sup_x = self.avg_pool_op(self.sup_x)

            sup_nshot = self.sup_x.shape[0]

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            # sup_y_g = F.avg_pool2d(self.sup_y, val_wsize) if isval else self.avg_pool_op(self.sup_y)
            sup_y_g = self.avg_pool_op(self.sup_y)

            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]

            glb_proto = torch.sum(self.sup_x * self.sup_y, dim=(-1, -2)) \
                        / (self.sup_y.sum(dim=(-1, -2)) + 1e-5)

            protos = torch.cat([protos, glb_proto], dim=0)

            return protos

        else:
            raise NotImplementedError

    def CalcDist(self,mode,protos,vis_sim = False):
        if mode == 'mask':
            pred_mask = F.cosine_similarity(self.qry, protos[..., None, None], dim=1, eps=1e-4) * 20.0  # [1, h, w]

            vis_dict = {'proto_assign': pred_mask}  # Nonethings to visualize
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict
        else:
            def safe_norm(x, p = 2, dim = 1, eps = 1e-4):
                x_norm = torch.norm(x, p = p, dim = dim) # .detach()
                x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
                x = x.div(x_norm.unsqueeze(1).expand_as(x))
                return x
            pro_n = safe_norm(protos)
            qry_n = safe_norm(self.qry)

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20

            pred_grid = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
            debug_assign = dists.argmax(dim=1).float().detach()

            vis_dict = {'proto_assign': debug_assign}  # things to visualize

            if vis_sim:  # return the similarity for visualization
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict

    def LVQ(self,protos_list,num_proto):
        cls = len(protos_list)
        assert cls== 2
        pro_labels = []
        for _id,_pro in enumerate(protos_list):
            pro_labels += [_id]*_pro.shape[0]
        pro_labels = np.array(pro_labels)
        pro_labels = pro_labels
        protos = torch.cat(protos_list,0)
        import random
        weights = []
        if isinstance(num_proto, int):
            num_proto = [num_proto]*cls
        _x = self.sup_x.view(self.sup_x.shape[0],self.sup_x.shape[1],-1).permute(0,2,1).squeeze(0)
        _y = self.sup_y.view(self.sup_y.shape[0], 1, -1).permute(1, 0, 2).view(1, -1).squeeze(0)
        for c in range(cls):
            _qry_c = _x[_y==c]
            ind_list = [i for i in range(_qry_c.shape[0])]
            random.shuffle(ind_list)
            num_proto[c] = min(num_proto[c],len(ind_list))
            ind_s = random.sample(ind_list,num_proto[c])
            weights.append(torch.cat((_qry_c[ind_s,:],(torch.tensor([c]*num_proto[c]).reshape(-1,1)).cuda()),1))
        weights = torch.cat(weights,0)
        weights = weights.detach().cpu().numpy()
        glvq = GlvqModel(prototypes_per_class=num_proto, initial_prototypes=weights)
        protos = protos.detach().cpu().numpy()
        glvq.fit(protos, pro_labels)
        bg_protos = glvq.w_[:num_proto[0],:]
        fg_protos = glvq.w_[num_proto[0]:,:]
        bg_protos = torch.FloatTensor(bg_protos)
        fg_protos = torch.FloatTensor(fg_protos)
        return bg_protos.cuda(),fg_protos.cuda()

    def LVQ_block(self,data_list,init_weights_list,num_proto,max_iter):
        # data_list: a list contains two components, ie, fore feature and back feature
        # init_weight_list:  a list contains two components, ie, fore candidate weights & back candidate feature
        # data processing
        cls = len(data_list)
        assert cls== 2
        data_labels = []
        for _id,_pro in enumerate(data_list):
            data_labels += [_id]*_pro.shape[0]
        data_labels = np.array(data_labels)
        data = torch.cat(data_list, 0)

        #initial weights propocessing
        weights = torch.cat(init_weights_list, 0)
        weights = weights.detach().cpu().numpy()

        #LVQ processing
        # print('LVQ block!!')
        # import pdb
        # pdb.set_trace()
        glvq = GlvqModel(prototypes_per_class=num_proto, initial_prototypes=weights, max_iter=max_iter) #
        data = data.detach().cpu().numpy()
        glvq.fit(data, data_labels)
        bg_protos = glvq.w_[:num_proto[0], :]
        fg_protos = glvq.w_[num_proto[0]:, :]
        bg_protos = torch.FloatTensor(bg_protos)
        fg_protos = torch.FloatTensor(fg_protos)
        return bg_protos.cuda(), fg_protos.cuda()

    def LVQ_feat_for_proto(self,proto_list,num_proto,max_iter):
        # print('feat_for_proto!!')
        # generate data_list: protos
        cls = 2
        # generate init_weight_list: feats
        if isinstance(num_proto, int):
            num_proto = [num_proto] * cls
        init_weight_list = []
        import random
        _x = self.sup_x.view(self.sup_x.shape[0],self.sup_x.shape[1],-1).permute(0,2,1).squeeze(0)
        _y = self.sup_y.view(self.sup_y.shape[0], 1, -1).permute(1, 0, 2).view(1, -1).squeeze(0)
        for c in range(2):
            _qry_c = _x[_y == c]
            ind_list = [i for i in range(_qry_c.shape[0])]
            random.shuffle(ind_list)
            num_proto[c] = min(num_proto[c], len(ind_list))
            ind_s = random.sample(ind_list, num_proto[c])
            init_weight_list.append(torch.cat((_qry_c[ind_s, :], (torch.tensor([c] * num_proto[c]).reshape(-1, 1)).cuda()), 1))
        return self.LVQ_block(proto_list,init_weight_list,num_proto,max_iter)

    def LVQ_proto_for_feat(self,proto_list,num_proto,max_iter):
        cls = 2
        # generate data_list: feats
        feats = []
        _x = self.sup_x.view(self.sup_x.shape[0],self.sup_x.shape[1],-1).permute(0,2,1).squeeze(0)
        _y = self.sup_y.view(self.sup_y.shape[0], 1, -1).permute(1, 0, 2).view(1, -1).squeeze(0)
        for c in range(2):
            _qry_c = _x[_y == c]
            feats.append(_qry_c)
        # generate init_weight_list: protos
        if isinstance(num_proto, int):
            num_proto = [num_proto] * cls
        init_weight_list = []
        import random
        for c in range(2):
            _proto_c = proto_list[c]
            ind_list = [i for i in range(_proto_c.shape[0])]
            random.shuffle(ind_list)
            num_proto[c] = min(num_proto[c], len(ind_list))
            ind_s = random.sample(ind_list, num_proto[c])
            init_weight_list.append(torch.cat((_proto_c[ind_s, :], (torch.tensor([c] * num_proto[c]).reshape(-1, 1)).cuda()), 1))
        return self.LVQ_block(feats,init_weight_list,num_proto,max_iter)

    def LVQ_feat_for_feat(self,proto_list,num_proto,max_iter):
        cls = 2
        # generate data_list: feats
        feats = []
        _x = self.sup_x.view(self.sup_x.shape[0], self.sup_x.shape[1], -1).permute(0, 2, 1).squeeze(0)
        _y = self.sup_y.view(self.sup_y.shape[0], 1, -1).permute(1, 0, 2).view(1, -1).squeeze(0)
        for c in range(2):
            _qry_c = _x[_y == c]
            feats.append(_qry_c)
        # generate init_weight_list: feats
        if isinstance(num_proto, int):
            num_proto = [num_proto] * cls
        init_weight_list = []
        import random
        for c in range(2):
            _qry_c = _x[_y == c]
            ind_list = [i for i in range(_qry_c.shape[0])]
            random.shuffle(ind_list)
            num_proto[c] = min(num_proto[c], len(ind_list))
            ind_s = random.sample(ind_list, num_proto[c])
            init_weight_list.append(torch.cat((_qry_c[ind_s, :], (torch.tensor([c] * num_proto[c]).reshape(-1, 1)).cuda()), 1))
        return self.LVQ_block(feats,init_weight_list,num_proto,max_iter)



    def SOM_labeled(self,nerous):
        from minisom import MiniSom
        _x = self.sup_x.view(self.sup_x.shape[0], self.sup_x.shape[1], -1).permute(0,2,1).squeeze(0)
        _y = self.sup_y.view(self.sup_y.shape[0], 1, -1).permute(1, 0, 2).view(1, -1).squeeze(0)
        _x = _x.detach().cpu()
        _y = _y.detach().cpu()
        # pixels = np.reshape(_x, (_x.shape[0] * _x.shape[1], -1))
        som = MiniSom(nerous, nerous, _x.shape[-1], sigma=1.,
                      learning_rate=0.2, neighborhood_function='bubble')
        som.random_weights_init(_x)
        som.train(_x, 10000, random_order=True, verbose=False)
        qnt = som.quantization(_x)
        FG_nerou = qnt[_y == 1,:]
        BG_nerou = qnt[_y == 0,:]
        FG_nerou = torch.FloatTensor(FG_nerou)
        BG_nerou = torch.FloatTensor(BG_nerou)
        return BG_nerou.cuda(), FG_nerou.cuda()

    def SOM_labeledV2(self,nerous,iter=10000,exclude=False, BGleft = False):
        from minisom import MiniSom
        _x = self.sup_x.view(self.sup_x.shape[0], self.sup_x.shape[1], -1).permute(0,2,1).squeeze(0)
        _y = self.sup_y.view(self.sup_y.shape[0], 1, -1).permute(1, 0, 2).view(1, -1).squeeze(0)
        _x = _x.detach().cpu()
        _y = _y.detach().cpu()
        som = MiniSom(nerous, nerous, _x.shape[-1], sigma=1.,
                      learning_rate=0.1, neighborhood_function='bubble')
        som.random_weights_init(_x)
        som.train(_x, iter, random_order=True, verbose=False)

        # import pdb
        # pdb.set_trace()
        _y = _y.numpy()
        _x = _x.numpy()
        FG_position = []
        BG_position = []
        map = np.argmin(som._distance_from_weights(_x),axis=1)
        for map_id in range(nerous*nerous):
            map_area = np.sum(np.uint8(map==map_id))
            _y_area = np.sum(_y[map==map_id])
            ratio = _y_area/map_area
            if ratio >= 0.95:
                FG_position.append(map_id)
            else:
                BG_position.append(map_id)
        #
        # map = np.argmin(som._distance_from_weights(_x), axis=1)
        # BG_position = map[_y == 0]
        # FG_position = map[_y == 1]
        # FG_position = np.unique(FG_position)
        # BG_position = np.unique(BG_position)
        if exclude:
            same_position = np.intersect1d(FG_position,BG_position)
            if BGleft:
                FG_position = np.setdiff1d(FG_position,same_position)
            else:
                BG_position = np.setdiff1d(BG_position,same_position)
        if len(FG_position) > 0:
            FG_nerou = som.get_weights()[np.unravel_index(FG_position,(nerous,nerous))]
        else:
            FG_nerou = _x[_y==1][0,:]
            FG_nerou = np.array([FG_nerou])
        BG_nerou = som.get_weights()[np.unravel_index(BG_position,(nerous,nerous))]
        FG_nerou = torch.FloatTensor(FG_nerou)
        BG_nerou = torch.FloatTensor(BG_nerou)
        # map = torch.from_numpy(map)
        return BG_nerou.cuda(), FG_nerou.cuda(), map



class Allocate(nn.Module):
    def __init__(self, weight_func, k=3, p=2):
        if weight_func=='inverse':
            self.weight_func = self.inverseweight
        elif weight_func=='gaussian':
            self.weight_func = self.gaussian
        elif weight_func=='subtra':
            self.weight_func = self.subtractweight
        else:
            self.weight_func = self.inverseweight
        self.k = k
        self.p = p

    def inverseweight(self, dist, num=1.0, const=0.1):
        # return torch.tensor(num).cuda() / (dist + torch.tensor(const).cuda())
        return num / (dist + const)

    def gaussian(self, dist, sigma=10.0):
        return math.e ** (- dist ** 2 / (2 * sigma ** 2))

    def subtractweight(self, dist, const=2.0):
        if dist > const:
            return 0.001
        else:
            return const - dist

    def find_knn(self,x,data2):
        knn_list = []
        for i in range(self.k):
            # import pdb
            # pdb.set_trace()
            dist = torch.dist(x, data2[i, :], p=self.p)
            knn_list.append((dist, i))
        for i in range(self.k, data2.shape[0]):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = torch.dist(x, data2[i, :], p=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, i)
        self.knn_list = knn_list

    def weighted_knn(self, data2):
        avg = 0.
        totalweight = 0.
        for i in range(len(self.knn_list)):
            weight = self.weight_func(self.knn_list[i][0])
            avg += data2[self.knn_list[i][1], :] * weight
            totalweight += weight
        self.avg = avg / totalweight

    def cal(self,data1,data2):
        data1_new = []
        for i in range(data1.shape[0]):
            x = data1[i, :]
            self.find_knn(x, data2)
            self.weighted_knn(data2)
            data1_new.append(self.avg)
        data1_new = torch.stack(data1_new)
        assert data1_new.shape == data1.shape
        return data1_new









