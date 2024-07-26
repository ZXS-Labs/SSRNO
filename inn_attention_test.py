import argparse
import os
import math
from functools import partial
import yaml
import torch
from torch.utils.data import DataLoader
from super_image import EdsrConfig,EdsrModel
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import datasets
import models
import utils
import onnxruntime as ort
import onnx
import torch_integral as inn
import numpy as np
def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell)
            ql = qr
            preds.append(pred)
        pred = torch.cat(preds, dim=2)
    return pred

def calc_psnr(pd,gt,dtr):
    pd = (pd[0][0]+pd[0][1]+pd[0][2])/3
    return peak_signal_noise_ratio((gt[0][0]*dtr).cpu().numpy(), (pd*dtr).cpu().numpy(), data_range=dtr.cpu().numpy())
def calc_ssim(pd,gt,dtr):
    pd = (pd[0][0]+pd[0][1]+pd[0][2])/3
    return ssim((gt[0][0]*dtr).cpu().numpy(), (pd*dtr).cpu().numpy(), data_range=dtr.cpu().numpy())

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, scale_max=4,
              verbose=False,mcell=False):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    elif eval_type.startswith('ixi'):
        scale = int(eval_type.split('-')[1])
        metric_fn = calc_psnr
    else:
        raise NotImplementedError

    val_res = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='val')
    cnt = 0
    for batch in pbar:
        cnt+=1

        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div
        coord = batch['coord']
        cell = batch['cell']
        if mcell == False: c = 1
        else : c = max(scale/scale_max, 1)

        if eval_bsize is None:
            with torch.no_grad():
                # from thop import profile
                # macs, params = profile(model,(inp,coord,cell*c))
                # print(macs*2/1e9,params/1e6)
                # exit()
 
                # 导入 Onnx 模型
                # Model = onnx.load('onnx/ssrno97.onnx')
                # onnx.checker.check_model(Model) # 验证Onnx模型是否准确
                # # 使用 onnxruntime 推理
                # model = ort.InferenceSession('onnx/ssrno97.onnx', providers=['CPUExecutionProvider'])
                # ce = cell*c
                # ce = ce.cpu().numpy()
                # inp = inp.cpu().numpy()
                # coord = coord.cpu().numpy()
                # start = time.time()
                # for i in range(100):
                #     output = model.run(['output'], {'inp':inp,'coord':coord,'cell':ce})
                # end = time.time()
                # print("time:",utils.time_text(end-start))
                # exit()
                # pred = torch.from_numpy(output[0])
                # pred = pred.cuda()

                
                # model = model.cpu()
                # inp = inp.cpu()
                # coord = coord.cpu()
                # ce = cell*c
                # ce = ce.cpu()

                # start = time.time()
                # for i in range(100):
                #     pred = model(inp, coord, cell*c)
                # end = time.time()
                # print("time:",utils.time_text(end-start))
                # exit()

                # torch.onnx.export(model,
                #                   (inp,coord,cell*c),
                #                   "onnx/ssrno97.onnx",
                #                   input_names=['inp','coord','cell'],
                #                   output_names=['output'],
                #                   dynamic_axes={
                #                     'inp': {0: 'batch_size', 2: 'input_height', 3: 'input_width'},
                #                     'coord': {0: 'batch_size',1: 'out_h', 2: 'out_w'},
                #                     'output': {0: 'batch_size', 2: 'out_h', 3: 'out_w'}
                #                     },
                #                   export_params=True)
                # exit()
                # with torch.profiler.profile(
                #     activities=[
                #         torch.profiler.ProfilerActivity.CPU,
                #         torch.profiler.ProfilerActivity.CUDA,
                #     ],
                #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/auto_attention_inplace'),
                #     profile_memory=True,
                #     record_shapes=True,
                #     with_stack=True
                # ) as profiler:
                #     for times in range(100):
                #         pred = model(inp, coord, cell*c)
                #         profiler.step()
                # exit()
                pred = model(inp, coord, cell*c)
        else:
            pred = batched_predict(model, inp, coord, cell*c, eval_bsize)

        with torch.no_grad():
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)
            res = metric_fn(pred, batch['gt'])
            #res = metric_fn(pred, batch['gt'], loader.dataset.dataset.dtr[(cnt-1)//96])

        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./configs/test_srno.yaml')
    parser.add_argument('--model')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--gpu', default='0,1')
    parser.add_argument('--mcell', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #print(os.environ['CUDA_VISIBLE_DEVICES'])

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_spec = torch.load(args.model)['model']

    model = models.make(model_spec, load_sd=False)
    if config["inn_attention"]:
        model.attentions.conv0.core.add_core_proj()
        model.attentions.conv1.core.add_core_proj()
    model = model.cuda()
    if config["inn_encoder"]:
        print("encoder changed")
        wrapper = inn.IntegralWrapper(init_from_discrete=False,permutation_config={"class":inn.permutation.NOptOutFiltersPermutation})
        dis_dims = {
            "head.0.weight": [0,1],
            "head.0.bias": [0],
            "body.15.body.2.weight":[0,1],
            "body.15.body.2.bias":[0]
        }
        model.encoder = wrapper(model.encoder, (1,3,128,128), inn.standard_continuous_dims(model.encoder), dis_dims)
        # for i in model.encoder.groups:
        #     if "operator" not in i.operations:
        #         size = i.size/2
        #     else:
        #         size = i.size
        #     i.reset_distribution(inn.UniformDistribution(int(i.size/2), int(i.size), i.base))
    if config["inn_attention"]:
        print("attentions changed")
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
        wrapper = inn.IntegralWrapper(init_from_discrete=False,permutation_config={"class":inn.permutation.NOptOutFiltersPermutation})
        model.attentions = wrapper(tmp, (1, 266, 64, 64), c_dims, dis_dims)
        # for i in model.attentions.groups:
        #     i.reset_distribution(inn.UniformDistribution(int(i.size/2), int(i.size), i.base))
    model.load_state_dict(model_spec['sd'])
    # if config['inn_encoder']:
    #     for i in model.encoder.groups:
    #         if i.size<4:
    #             size = i.size
    #         else:
    #             size = i.size/2
    #         i.resize(int(size))
    # if config["inn_attention"]:
    #     for i in model.attentions.groups:
    #         if i.typeo!="max":
    #             i.resize(int(i.size/2))
    
    if config["inn_encoder"]:
        model.encoder.threshold = 0.99
        model.encoder.set_rate()
        model.encoder.set_distributions()
        model.encoder.set_size()
    if config["inn_attention"]:
        model.attentions.threshold = 0.93
        model.attentions.set_rate()
        model.attentions.set_distributions()
        model.attentions.set_size()
    if config['inn_encoder']:
        print("Compression: ", model.encoder.eval().calculate_compression())
        model.encoder = model.encoder.get_unparametrized_model()
    if config["inn_attention"]:
        print("Compression: ", model.attentions.eval().calculate_compression()) 
        model.attentions = model.attentions.get_unparametrized_model()
    print(args.model)
    roots = ["./data/benchmark/Urban100/HR","./data/benchmark/B100/HR", "./data/benchmark/Set14/HR","./data/benchmark/Set5/HR"]
    scals = [4,6]
    evaltp = "benchmark"
    # roots = ["./data/DIV2K_valid_HR"]
    # scals = [2,3,4,6,12,18,24,30]
    # config['eval_bsize'] = 300
    # evaltp = "div2k"
    for rt in roots:
        for idx in range(len(scals)):
            config["test_dataset"]["dataset"]["args"]["root_path"] = rt
            config["test_dataset"]["wrapper"]["args"]["scale_min"] = scals[idx]
            config["test_dataset"]["wrapper"]["args"]["scale_max"] = scals[idx]
            config["eval_type"] = evaltp+"-"+str(scals[idx])

            spec = config['test_dataset']
            dataset = datasets.make(spec['dataset'])
            dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
            loader = DataLoader(dataset, batch_size=spec['batch_size'],
                num_workers=8, pin_memory=True, shuffle=False)
            
            import time
            t1= time.time()
            res = eval_psnr(loader, model,
                data_norm=config.get('data_norm'),
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'),
                scale_max = int(args.scale_max),
                verbose=True,
                mcell=bool(args.mcell))
            t2 =time.time()
            # print('result: {:.4f}'.format(res), utils.time_text(t2-t1))
            # print('model: #params={}'.format(utils.compute_num_params(model, text=True)))
            print(rt,"time:",utils.time_text(t2-t1),"x{}".format(scals[idx]),'result: {:.4f}'.format(res),'model: #params={}'.format(int(sum([np.prod(p.shape) for p in model.parameters()]))))
    
