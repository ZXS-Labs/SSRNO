import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
import datasets
import models
import utils
from test import eval_psnr
from scheduler import GradualWarmupScheduler

import torch_integral as inn

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True,persistent_workers=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    return train_loader


def prepare_training():
    sv_file = torch.load(config['resume'],map_location=torch.device('cpu'))
    model = models.make(sv_file['model'], load_sd=True)
    model.attentions.conv0.core.add_core_proj()
    model.attentions.conv1.core.add_core_proj()
    model = model.cuda()
    #change encoder
    if config.get("inn_encoder"):
        print("encoder layers changed")
        tmp = model.encoder
        if config["model"]["args"]["encoder_spec"]["name"] == "rdn":
            dis_dims = {
                "SFENet1.weight":[0,1],
                "SFENet1.bias":[0],
                "GFF.1.weight":[0,1],
                "GFF.1.bias":[0]
            }
        else:
            dis_dims = {
                "head.0.weight": [0,1],
                "head.0.bias": [0],
                "body.15.body.2.weight":[0,1],
                "body.15.body.2.bias":[0]
            }
        model.encoder = inn.IntegralWrapper(
            init_from_discrete=True,
            permutation_config={"class":inn.permutation.NOptOutFiltersPermutation}
            )(tmp, (1, 3, 128, 128), inn.standard_continuous_dims(tmp), dis_dims)
        # for i in model.encoder.groups:
        #     if "operator" not in i.operations:
        #         size = i.size/2
        #     else:
        #         size = i.size
        #     i.reset_distribution(inn.UniformDistribution(int(size), int(i.size), i.base))
        model.encoder.set_rate()
        model.encoder.set_distributions()
    #change galerkin attention
    if config.get("inn_attention"):
        print("attention layer changed")
        tmp = model.attentions
        c_dims = inn.standard_continuous_dims(tmp)
        c_dims.update({"conv0.core.kln.weight":[0,1],
                    "conv0.core.kln.bias":[0,1],
                    "conv0.core.vln.weight":[0,1],
                    "conv0.core.vln.bias":[0,1],
                    "conv1.core.kln.weight":[0,1],
                    "conv1.core.kln.bias":[0,1],
                    "conv1.core.vln.weight":[0,1],
                    "conv1.core.vln.bias":[0,1]})
        dis_dims = {
            "conv00.weight":[1],
            "fc1.weight":[0],
            "fc1.bias":[0]
        }
        model.attentions = inn.IntegralWrapper(
            init_from_discrete=True,
            permutation_config={"class":inn.permutation.NOptOutFiltersPermutation}
            )(tmp, (1, 266, 64, 64), c_dims, dis_dims)
        # for i in model.attentions.groups:
        #     i.reset_distribution(inn.UniformDistribution(int(i.size/2), int(i.size), i.base, i.domain))
        model.attentions.set_rate()
        model.attentions.set_distributions()
    if config.get('continue'):
        model.load_state_dict(torch.load(config['continue_resume'],map_location=torch.device('cpu'))['model']['sd'])
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        # for e in range(1,epoch_start):
        #     lr_scheduler.step(e)
        #     #lr_scheduler.step()
    else:
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('exp_lr') is not None:
            lr_scheduler = ExponentialLR(optimizer, gamma=config['exp_lr']['gamma'])
        elif config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, \
         epoch):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()
    metric_fn = utils.calc_psnr

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    
    #num_dataset = 800 # DIV2K
    #iter_per_epoch = int(num_dataset / config.get('train_dataset')['batch_size'] \
    #                    * config.get('train_dataset')['dataset']['args']['repeat'])
    iteration = 0
    pbar = tqdm(train_loader, leave=False, desc='train')
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div
        if config.get("inn_encoder"):
            if n_gpus > 1:
                model.module.encoder.forward_groups()
            else:
                model.encoder.forward_groups()
        if config.get("inn_attention"):
            if n_gpus > 1:
                model.module.attentions.forward_groups()
            else:
                model.attentions.forward_groups()
        pred = model(inp, batch['coord'], batch['cell'])
        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)
        #psnr = metric_fn(pred, gt)
        
        # tensorboard
        #writer.add_scalars('loss', {'train': loss.item()}, (epoch-1)*iter_per_epoch + iteration)
        #writer.add_scalars('psnr', {'train': psnr}, (epoch-1)*iter_per_epoch + iteration)
        iteration += 1
        
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None
        pbar.set_description('train {:.4f}'.format(train_loss.item()))
        
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, \
                           epoch)
        if lr_scheduler is not None:
            # lr_scheduler.step()
            lr_scheduler.step(epoch)

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if ((epoch_save is not None) and (epoch % epoch_save*2 == 0)):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch))) 
            if n_gpus > 1:
                model.module.attentions.set_rate()
                model.module.attentions.set_distributions()
            else:
                model.attentions.set_rate() 
                model.attentions.set_distributions()         
#         if (epoch == 1) or ((epoch_val is not None) and (epoch % epoch_val == 0)):
#             if n_gpus > 1: #and (config.get('eval_bsize') is not None):
#                 model_ = model.module
#             else:
#                 model_ = model
#             val_res = eval_psnr(val_loader, model_,
#                 data_norm=config['data_norm'],
#                 eval_type=config.get('eval_type'),
#                 eval_bsize=config.get('eval_bsize'))

#             log_info.append('val: psnr={:.4f}'.format(val_res))
# #             writer.add_scalars('psnr', {'val': val_res}, epoch)
#             if val_res > max_val_v:
#                 max_val_v = val_res
#                 torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./configs/train_edsr-sronet.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)