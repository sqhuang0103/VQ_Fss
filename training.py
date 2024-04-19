"""
Training the model
Extended from original implementation of PANet by Wang et al.
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np


from model.grid_proto_fewshot import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO
from dataloaders.dataset_utils import get_normalize_op
import dataloaders.augutils as myaug
from dataloaders.dev_customized_med import med_fewshot_val

from util.utils import set_seed, t2n, to01, compose_wt_simple
from util.metric import Metric
from util.seed_init import generate_s_seed
from config_ssl_upload import ex
import tqdm


# from ASG_model import *
# from PFE_model.PFENet import PFENet
# net = 'pfenet' #'' asgnet
# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"


def eval(_log,_config,model,test_labels,te_dataset,te_parent,testloader,metric_node):
    _log.info('###### Starting validation ######')
    # net = _config["net_name"]
    # print('====Employ Net:'+net+'=======')
    model.eval()
    metric_node.reset()


    with torch.no_grad():
        for curr_lb in test_labels:
            te_dataset.set_curr_cls(curr_lb)
            te_support_batched = te_parent.get_support(curr_class=curr_lb, class_idx=[curr_lb],
                                                    scan_idx=_config["support_idx"], npart=_config['task']['npart'])

            # way(1 for now) x part x shot x 3 x H x W] #
            te_support_images = [[shot.cuda() for shot in way]
                              for way in te_support_batched['support_images']]  # way x part x [shot x C x H x W]
            suffix = 'mask'
            te_support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                               for way in te_support_batched['support_mask']]
            te_support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                               for way in te_support_batched['support_mask']]


            curr_scan_count = -1  # counting for current scan

            for te_sample_batched in testloader:

                _scan_id = te_sample_batched["scan_id"][0]  # we assume batch size for query is 1
                if _scan_id in te_parent.potential_support_sid:  # skip the support scan, don't include that to query
                    continue
                if te_sample_batched["is_start"]:
                    ii = 0
                    curr_scan_count += 1
                    _scan_id = te_sample_batched["scan_id"][0]
                    outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                    outsize = (256, 256, outsize[0])  # original image read by itk: Z, H, W, in prediction we use H, W, Z
                    # _pred = np.zeros(outsize)
                    # _pred.fill(np.nan)

                q_part = te_sample_batched["part_assign"]  # the chunck of query, for assignment with support
                te_query_images = [te_sample_batched['image'].cuda()]
                te_query_labels = torch.cat([te_sample_batched['label'].cuda()], dim=0)

                # [way, [part, [shot x C x H x W]]] ->
                sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in
                                 te_support_images[0][q_part]]]  # way(1) x shot x [B(1) x C x H x W]
                sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in te_support_fg_mask[0][q_part]]]
                sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in te_support_bg_mask[0][q_part]]]

                te_query_pred, _, _, _ = model(sup_img_part, sup_fgm_part, sup_bgm_part, te_query_images,
                                                      isval=True, val_wsize=_config["val_wsize"])
                te_query_pred = np.array(te_query_pred.argmax(dim=1)[0].cpu())

                # te_query_pred = model(s_x=sup_img_part, s_y=sup_fgm_part, x=te_query_images, y=te_query_labels, s_seed=te_s_init_seed)

                # te_query_pred = np.array(te_query_pred.argmax(dim=1)[0].cpu())
                # _pred[..., ii] = te_query_pred.copy()
                if (te_sample_batched["z_id"] - te_sample_batched["z_max"] <= _config['z_margin']) and (
                        te_sample_batched["z_id"] - te_sample_batched["z_min"] >= -1 * _config['z_margin']):
                    metric_node.record(te_query_pred, np.array(te_query_labels[0].cpu()), labels=[curr_lb],
                                               n_scan=curr_scan_count)
                else:
                    pass

                ii += 1

    # del te_sample_batched, te_support_images, te_support_bg_mask, te_query_images, te_query_labels, te_query_pred
    # del

    # compute dice scores by scan

    classDice,_, meanDice,_, m_rawDice = metric_node.get_mDice(labels=sorted(test_labels), n_scan=None,
                                                                             give_raw=True)

    metric_node.reset()  # reset this calculation node
    model.train()
    _log.info(f'###### End of validation ######')
    return classDice,meanDice

@ex.automain
def main(_run, _config, _log):

    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            #_run.observers[0].dir:'/data3/huangsq/exp_MR/CHAOST2_Superpix_nametest_lbset0_set2/SSL_train_vfold0_1shot/3'
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)
    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=None, cfg=_config['model'])
    # model = nn.DataParallel(model)
    model = model.cuda()
    model.train()

    ### Training set
    data_name = _config['dataset']
    _log.info(f'###### Load dataset {data_name} ######')
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
    elif data_name == 'C0_Superpix':
        # raise NotImplementedError
        baseset_name = 'C0'
        train_labels = 'pseudo labels'
        test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all']
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
        train_labels = 'pseudo labels'
        test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all']
    elif data_name == 'CHAOST2':
        baseset_name = 'CHAOST2'
        train_labels = DATASET_INFO[baseset_name]['LABEL_GROUP'][
            'pa_all']  #DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]] # - set([1,4]) #DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all']
        test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    elif data_name == 'Pros_Superpix':
        baseset_name = 'Pros'
        train_labels = 'pseudo labels'
        test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all']
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    if baseset_name == 'SABS':  # for CT we need to know statistics of
        t_parent = SuperpixelDataset(  # base dataset
            which_dataset=baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split=_config['eval_fold'],
            mode='train',
            min_fg=str(_config["min_fg_data"]),  # dummy entry for superpixel dataset
            transforms=None,
            nsup=_config['task']['n_shots'],
            scan_per_load=_config['scan_per_load'],
            exclude_list=_config["exclude_cls_list"],
            superpix_scale=_config["superpix_scale"],
            fix_length=_config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (
                    data_name == 'CHAOST2_Superpix') else None
        )
        norm_func = t_parent.norm_func
    else:
        norm_func = get_normalize_op(modality='MR', fids=None)

    ### Transforms for data augmentation
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly
    # train_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] #- set([1,4]) #DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all']
    # test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Labels in training : {[[lb for lb in train_labels] if not isinstance(train_labels,str) else train_labels]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    if data_name in ['CHAOST2_Superpix', 'C0_Superpix', 'Pros_Superpix']:
        tr_parent = SuperpixelDataset( # base dataset
            which_dataset = baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split = _config['eval_fold'],
            mode='train',
            min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
            transforms=tr_transforms,
            nsup = _config['task']['n_shots'],
            scan_per_load = _config['scan_per_load'],
            exclude_list = _config["exclude_cls_list"],
            superpix_scale = _config["superpix_scale"],
            fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None
        )
    elif data_name == 'CHAOST2':
        tr_parent, tr_dataset = med_fewshot(
            dataset_name=baseset_name,
            base_dir=_config['path'][baseset_name]['data_dir'],
            idx_split=_config['eval_fold'],
            mode='train',
            min_fg=str(_config["min_fg_data"]),
            transforms=tr_transforms,
            scan_per_load=_config['scan_per_load'],
            exclude_list=_config["exclude_cls_list"],
            act_labels=train_labels, #DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]],
            npart=_config['task']['npart'],
            nsup=_config['task']['n_shots'],
            n_ways=_config['task']['n_ways'],
            n_shots=_config['task']['n_shots'],
            max_iters_per_load=_config['max_iters_per_load'],
            extern_normalize_func=norm_func
        )
    else:
        pass
    # if data_name == 'CHAOST2_Superpix':
    #     print('CHAOST2_Superpix')
    #     tr_parent = SuperpixelDataset(  # base dataset
    #         which_dataset=baseset_name,
    #         base_dir=_config['path'][data_name]['data_dir'],
    #         idx_split=_config['eval_fold'],
    #         mode='train',
    #         min_fg=str(_config["min_fg_data"]),  # dummy entry for superpixel dataset
    #         transforms=tr_transforms,
    #         nsup=_config['task']['n_shots'],
    #         scan_per_load=_config['scan_per_load'],
    #         exclude_list=_config["exclude_cls_list"],
    #         superpix_scale=_config["superpix_scale"],
    #         fix_length=_config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (
    #                     data_name == 'CHAOST2_Superpix') else None
    #     )


    ### dataloaders
    trainloader = DataLoader(
        tr_parent,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    # test dataloader!
    if _config['val_on_test']:
        _log.info('#######  load the testing data #######')
        if data_name == 'SABS_Superpix':
            baseset_name = 'SABS'
            max_label = 13
        elif data_name == 'C0_Superpix':
            # raise NotImplementedError
            baseset_name = 'C0'
            max_label = 3
        elif data_name == 'CHAOST2_Superpix':
            baseset_name = 'CHAOST2'
            max_label = 4
        elif data_name == 'CHAOST2':
            baseset_name = 'CHAOST2'
            max_label = 4
        elif data_name == 'Pros_Superpix':
            baseset_name = 'Pros'
            max_label = 8
        else:
            raise ValueError(f'Dataset: {data_name} not found')

        te_dataset, te_parent = med_fewshot_val(
            dataset_name=baseset_name,
            base_dir=_config['path'][baseset_name]['data_dir'],
            idx_split=_config['eval_fold'],
            scan_per_load=_config['scan_per_load'],
            act_labels=test_labels,
            npart=_config['task']['npart'],
            nsup=_config['task']['n_shots'],
            extern_normalize_func=norm_func
        )

        ### dataloaders
        testloader = DataLoader(
            te_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            drop_last=False
        )


        _log.info('###### Set validation nodes ######')
        metric_node = Metric(max_label=max_label,
                                     n_scans=len(te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'])
        # print(len(te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'])


    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    else:
        raise NotImplementedError

    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)

    i_iter = -1 # total number of iteration
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading

    log_loss = {'loss': 0, 'align_loss': 0}


    _log.info('###### Training ######')
    best_dice = [0.] *len(test_labels)
    best_mdice = 0.
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch+1} of {n_sub_epoches} epoches ######')
        for _, sample_batched in enumerate(trainloader):
            # Prepare input
            i_iter += 1
            # add writers
            support_images = [[shot.cuda() for shot in way]
                              for way in sample_batched['support_images']]
            support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]

            query_images = [query_image.cuda()
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

            query_pred, align_loss, debug_vis, assign_mats = model(support_images, support_fg_mask, support_bg_mask, query_images, isval = False, val_wsize = None)
            # print(query_pred.shape,query_labels.shape) # torch.Size([1, 2, 256, 256]) torch.Size([1, 256, 256])
            optimizer.zero_grad()

            # query_pred, align_loss, debug_vis, assign_mats = model(support_images, support_fg_mask, support_bg_mask, query_images, isval = False, val_wsize = None)
            # output, query_loss, align_loss = model(s_x=support_images, s_y=support_fg_mask, x=query_images, y=query_labels, s_seed=s_init_seed)

            query_loss = criterion(query_pred, query_labels)
            loss = query_loss + align_loss
            loss.mean().backward()
            optimizer.step()
            scheduler.step()


            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            _run.log_scalar('loss', query_loss)
            _run.log_scalar('align_loss', align_loss)
            log_loss['loss'] += query_loss
            log_loss['align_loss'] += align_loss

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:
                loss = log_loss['loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0

                print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
                torch.save(model.encoder.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'resnet_{i_iter + 1}.pth'))

            if _config['val_on_test'] and ((i_iter+1) % _config['eval_every'] == 0):
                classdice,meandice = eval(_log, _config, model, test_labels, te_dataset, te_parent, testloader, metric_node)
                print(f'#########classDice: {classdice}, meanDice: {meandice}####')
                if meandice > best_mdice:
                    best_mdice = meandice
                    torch.save(model.state_dict(),os.path.join(f'{_run.observers[0].dir}/snapshots', f'best.pth'))
                for _i in range(len(best_dice)):
                    if classdice[_i] > best_dice[_i]:
                        best_dice[_i] = classdice[_i]
                        torch.save(model.state_dict(),os.path.join(f'{_run.observers[0].dir}/snapshots', f'class_{list(test_labels)[_i]}_best.pth'))

            if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix' or data_name == 'Pros_Superpix':
                if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                    _log.info('###### Reloading dataset ######')
                    trainloader.dataset.reload_buffer()
                    print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

            if (i_iter - 2) > _config['n_steps']:
                return 1 # finish up



