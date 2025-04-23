import contextlib
import fire
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mup
import random
import math
import numpy as np
import logging
import sys
import lib.ddp
import lib.datasets
import lib.decay_to_init
import lib.ema
import lib.models
import lib.ops
import lib.utils
import os
import torch
import torch.distributed.optim
from torch import optim, autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.datasets import get_dataloader, get_dataloaders, infinite_loader
from evaluation_batch import evaluate, generate_samples

# 首先导入SelfCheck模块
from lib.selfcheck import SelfCheckVerifier

class SelfCheckConfig:
    def __init__(self, 
                enabled=False, 
                confidence_threshold=0.7, 
                weight=0.1, 
                apply_corrections=True):
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.weight = weight
        self.apply_corrections = apply_corrections

def train_loop_with_selfcheck(args, modules, tokenizer, train_iterator, train_mode=True):
    """
    修改后的训练循环，包含SelfCheck机制
    """
    # 创建SelfCheck验证器
    verifier = SelfCheckVerifier(tokenizer, modules, args)
    
    # 原有参数设置
    batch_size = args.batch_size
    reconst_weight = args.reconst_weight
    diffusion_weight = args.diffusion_weight
    reconst_bs_cache = {}
    
    # SelfCheck相关参数
    use_selfcheck = args.use_selfcheck
    confidence_threshold = args.confidence_threshold
    
    # 初始化训练状态
    reconst_ema = torch.tensor(1e-8).cuda()
    diffusion_ema = torch.tensor(1e-8).cuda()
    reconst_sqr_ema = torch.tensor(1e-8).cuda()
    diffusion_sqr_ema = torch.tensor(1e-8).cuda()
    loss_ema_bias = torch.tensor(1e-8).cuda()
    
    # 设置模型到训练模式
    for name, module in modules.items():
        module.train(train_mode)
    
    # 获取一个批次的数据
    x, src_mask, tgt_mask = next(train_iterator)
    x = x.cuda()
    src_mask = src_mask.cuda()
    tgt_mask = tgt_mask.cuda()
    
    # 计算reconst_bs (重建批次大小)
    if step in reconst_bs_cache:
        reconst_bs = reconst_bs_cache[step]
    else:
        # 原有的reconst_bs计算逻辑
        b = 1 / loss_ema_bias
        reconst_std = (b*reconst_sqr_ema - (b*reconst_ema)**2).clamp(min=0).sqrt()
        diffusion_std = (b*diffusion_sqr_ema - (b*diffusion_ema)**2).clamp(min=0).sqrt()
        reconst_bs = batch_size * (reconst_std / (1e-8 + reconst_std + diffusion_std))
        reconst_bs = int(reconst_bs.round().clamp(1, batch_size-1))
        reconst_bs_cache[step] = reconst_bs
    
    # 准备timesteps采样
    t = torch.empty([batch_size], device='cuda')
    t[:reconst_bs] = 0
    
    # 采样剩余的timesteps
    t[reconst_bs:] = torch.linspace(0, 1, batch_size-reconst_bs+1)[1:]
    
    # 噪声采样
    embedding_matrix = modules['embedding_matrix']()
    x_embed = embedding_matrix[x]
    noise = torch.randn_like(x_embed)
    
    # 扩散过程
    gamma = modules['noise_schedule'](t)
    gamma_0 = modules['noise_schedule'](torch.zeros_like(t))
    gamma_1 = modules['noise_schedule'](torch.ones_like(t))
    
    alpha_t = torch.sqrt((1 - gamma / gamma_1))
    sigma_t = torch.sqrt(gamma / gamma_1)
    
    # 添加噪声生成z
    z = alpha_t[:, None, None] * x_embed + sigma_t[:, None, None] * noise
    
    # 进行自我条件采样
    x_selfcond = torch.zeros_like(x_embed, device='cuda', dtype=torch.float32)
    
    # ===== SelfCheck修改点：在使用模型前进行步骤验证和筛选 =====
    if use_selfcheck and args.cot:
        # 提取每个样本的问题和思考步骤
        questions = []
        all_steps = []
        
        for i in range(batch_size):
            text = tokenizer.decode(x[i].tolist())
            parts = text.split(lib.datasets.SEP_TOKEN)
            
            if len(parts) > 1:
                # 假设第一部分是问题
                question = parts[0].strip()
                # 剩余部分是思考步骤
                steps = [p.strip() for p in parts[1:] if p.strip()]
                
                questions.append(question)
                all_steps.append(steps)
            else:
                # 如果没有分隔符，整个文本作为问题
                questions.append(text)
                all_steps.append([])
        
        # 对需要进行推理的样本（t>0）进行SelfCheck
        for i in range(reconst_bs, batch_size):
            if len(all_steps[i]) > 0:
                # 对最后一个步骤进行验证
                last_step = all_steps[i][-1]
                previous_steps = all_steps[i][:-1]
                
                try:
                    # 进行验证
                    confidence, verified_step = verifier.verify_step(
                        questions[i],
                        last_step,
                        previous_steps,
                        len(previous_steps)
                    )
                    
                    # 如果置信度低，使用修正的步骤
                    if confidence < confidence_threshold:
                        # 替换最后一个步骤
                        all_steps[i][-1] = verified_step
                        
                        # 重新构建输入
                        new_text = questions[i] + lib.datasets.SEP_TOKEN
                        new_text += lib.datasets.SEP_TOKEN.join(all_steps[i])
                        
                        # 重新编码
                        new_tokens = tokenizer.encode(new_text)[:x.shape[1]]
                        # 如果长度不足，填充
                        if len(new_tokens) < x.shape[1]:
                            padding = [lib.datasets.PAD_TOKEN_ID] * (x.shape[1] - len(new_tokens))
                            new_tokens.extend(padding)
                        
                        # 更新x
                        x[i] = torch.tensor(new_tokens, device=x.device)
                        
                        # 更新embedding
                        x_embed[i] = embedding_matrix[x[i]]
                        
                        # 重新计算z
                        z[i] = alpha_t[i] * x_embed[i] + sigma_t[i] * noise[i]
                except Exception as e:
                    # 出错时记录但继续执行
                    print(f"SelfCheck error: {e}")
    
    # 设置bias scale
    if train_mode:
        bias_scale = min(1., (step + 1e-8) / (args.bias_warmup_steps + 1e-8))
    else:
        bias_scale = 1.
    
    # 模型前向传播
    # 使用自我条件
    logits_selfcond, x_reconst_selfcond = modules['model'](
        z_selfcond=z, 
        gamma_selfcond=gamma, 
        embedding_matrix=embedding_matrix, 
        bias_scale=bias_scale,
        x_selfcond=x_selfcond,
        x_embed=x_embed if args.fix_src else None,
        src_mask=src_mask if args.fix_src else None
    )
    
    # ===== SelfCheck修改点：在训练损失中添加SelfCheck损失 =====
    # 计算原始损失
    reconst_loss = F.cross_entropy(
        logits_selfcond.reshape(-1, logits_selfcond.shape[-1]),
        x.reshape(-1),
        reduction='none'
    ).reshape(x.shape)
    
    reconst_loss_masked = (reconst_loss * tgt_mask).sum(-1)
    reconst_loss = (reconst_loss_masked / tgt_mask.sum(-1)).mean()
    
    # 计算扩散损失
    eps_pred = noise - (z - alpha_t[:, None, None] * x_reconst_selfcond) / sigma_t[:, None, None]
    diffusion_loss = (eps_pred ** 2).mean()
    
    # 如果使用SelfCheck，添加SelfCheck损失
    if use_selfcheck and args.cot:
        # 创建SelfCheck损失
        selfcheck_loss = 0.0
        
        # 获取模型预测
        predicted_steps = []
        for i in range(reconst_bs, batch_size):
            pred_tokens = logits_selfcond[i].argmax(dim=-1)
            pred_text = tokenizer.decode(pred_tokens.tolist())
            predicted_steps.append(pred_text)
        
        # 针对每个样本计算SelfCheck损失
        selfcheck_losses = []
        for i, (pred, truth) in enumerate(zip(predicted_steps, all_steps[reconst_bs:])):
            # 略过没有步骤的样本
            if not truth:
                continue
                
            # 提取预测的最后一个步骤
            pred_parts = pred.split(lib.datasets.SEP_TOKEN)
            if len(pred_parts) > 1:
                pred_last_step = pred_parts[-1].strip()
            else:
                pred_last_step = pred
                
            # 提取真实的最后一个步骤
            truth_last_step = truth[-1]
            
            try:
                # 使用验证器计算置信度
                confidence, _ = verifier.compare_steps(
                    pred_last_step,
                    truth_last_step,
                    questions[i + reconst_bs],
                    "完成推理" # 简化的目标描述
                )
                
                # 将置信度转换为损失
                # 低置信度对应高损失
                step_loss = 1.0 - confidence
                selfcheck_losses.append(step_loss)
            except Exception as e:
                print(f"SelfCheck loss error: {e}")
        
        # 计算平均SelfCheck损失
        if selfcheck_losses:
            selfcheck_loss = torch.tensor(sum(selfcheck_losses) / len(selfcheck_losses),
                                         device=reconst_loss.device)
        else:
            selfcheck_loss = torch.tensor(0.0, device=reconst_loss.device)
        
        # 添加到总损失
        selfcheck_weight = args.selfcheck_weight  # 新参数：SelfCheck损失权重
        total_loss = reconst_weight * reconst_loss + diffusion_weight * diffusion_loss + selfcheck_weight * selfcheck_loss
    else:
        # 原始损失计算
        total_loss = reconst_weight * reconst_loss + diffusion_weight * diffusion_loss
    
    # 更新模型参数
    total_loss.backward()
    
    # 更新EMA统计
    if train_mode:
        with torch.no_grad():
            loss_ema_bias.lerp_(torch.tensor(1., device='cuda'), 1 - args.reconst_bs_ema)
            reconst_ema.lerp_((args.reconst_weight * reconst_loss).sum() / avg_reconst_bs, 1 - args.reconst_bs_ema)
            reconst_sqr_ema.lerp_((args.reconst_weight * reconst_loss).pow(2).sum() / avg_reconst_bs, 1 - args.reconst_bs_ema)
            diffusion_ema.lerp_((args.diffusion_weight * diffusion_loss).sum() / avg_diffusion_bs, 1 - args.reconst_bs_ema)
            diffusion_sqr_ema.lerp_((args.diffusion_weight * diffusion_loss).pow(2).sum() / avg_diffusion_bs, 1 - args.reconst_bs_ema)
    
    return total_loss

def masked_loss(loss, mask, weight, dim=None):
    loss = loss.masked_fill(~mask, 0)
    loss = loss * weight
    average_loss = loss.sum(dim)/(mask.sum(dim)+0.01)
    return average_loss

def set_args(args):
    bs = lib.ddp.world_size()*args.batch_size*args.grad_accum_steps
    save_weights_path=f"outputs/{args.dataset}-bs{bs}"
    if args.fix_src:
        save_weights_path += '-fix_src'
    if args.cot:
        save_weights_path += '-cot'
    if args.digit:
        save_weights_path += '-digit'
    args.save_weights_path = save_weights_path + f'-steps{args.steps}'

    if lib.ddp.rank() == 0:
        os.makedirs(args.save_weights_path, exist_ok=True)
        args.train_log = os.path.join(args.save_weights_path, "train.log")
        if os.path.exists(args.train_log): 
            os.remove(args.train_log)

        targets = logging.StreamHandler(sys.stdout), logging.FileHandler(args.train_log, mode='w')
        logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO, handlers=targets)


def sampling_gold_prob(i, steps, min_prob=0.1):
    return (1-min_prob)*(steps-i)/steps + min_prob


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 在train.py中修改
def forward_with_selfcheck(step, idx_in_batch, batch_size, tokenizer=None, selfcheck_verifier=None, selfcheck_config=None):
    """
    支持SelfCheck的前向传播函数
    """
    # 获取数据
    x, src_mask, tgt_mask = next(train_iterator)
    x = x.cuda()
    src_mask = src_mask.cuda()
    tgt_mask = tgt_mask.cuda()
    
    # 原始处理逻辑
    batch_size = x.shape[0]
    reconst_bs = args.reconst_bs
    
    # Diffusion过程与原始代码相同
    embedding_matrix = modules['embedding_matrix']()
    x_embed = embedding_matrix[x]
    noise = torch.randn_like(x_embed)
    
    # 时间步和噪声参数与原始代码相同
    t = torch.empty([batch_size], device='cuda')
    t[:reconst_bs] = 0
    t[reconst_bs:] = torch.linspace(0, 1, batch_size-reconst_bs+1)[1:]
    
    gamma = modules['noise_schedule'](t)
    gamma_0 = modules['noise_schedule'](torch.zeros_like(t))
    gamma_1 = modules['noise_schedule'](torch.ones_like(t))
    
    alpha_t = torch.sqrt((1 - gamma / gamma_1))
    sigma_t = torch.sqrt(gamma / gamma_1)
    
    z = alpha_t[:, None, None] * x_embed + sigma_t[:, None, None] * noise
    
    # 自我条件化参数
    x_selfcond = torch.zeros_like(x_embed, device='cuda', dtype=torch.float32)
    
    # ===== SelfCheck逻辑 =====
    selfcheck_loss = torch.tensor(0.0, device='cuda')
    
    if selfcheck_config and selfcheck_config.enabled and args.cot and selfcheck_verifier and tokenizer:
        # 提取问题和思考步骤
        questions = []
        all_steps = []
        
        for i in range(batch_size):
            text = tokenizer.decode(x[i].tolist())
            parts = text.split(lib.datasets.SEP_TOKEN)
            
            if len(parts) > 1:
                # 假设第一部分是问题
                question = parts[0].strip()
                # 剩余部分是思考步骤
                steps = [p.strip() for p in parts[1:] if p.strip()]
                
                questions.append(question)
                all_steps.append(steps)
            else:
                # 如果没有分隔符，整个文本作为问题
                questions.append(text)
                all_steps.append([])
        
        # 对需要进行推理的样本(t>0)应用SelfCheck
        modified = False
        for i in range(reconst_bs, batch_size):
            if len(all_steps[i]) > 0:
                # 对最后一个步骤进行验证
                last_step = all_steps[i][-1]
                previous_steps = all_steps[i][:-1]
                
                try:
                    # 进行验证
                    confidence, verified_step = selfcheck_verifier.verify_step(
                        questions[i],
                        last_step,
                        previous_steps,
                        len(previous_steps)
                    )
                    
                    # 计算SelfCheck损失（1 - 置信度）
                    step_loss = 1.0 - confidence
                    selfcheck_loss = selfcheck_loss + step_loss
                    
                    # 如果置信度低且需要修正
                    if confidence < selfcheck_config.confidence_threshold and selfcheck_config.apply_corrections:
                        # 替换最后一个步骤
                        all_steps[i][-1] = verified_step
                        modified = True
                        
                except Exception as e:
                    logging.warning(f"SelfCheck error: {e}")
        
        # 如果有修改，更新输入
        if modified:
            for i in range(reconst_bs, batch_size):
                if len(all_steps[i]) > 0:
                    # 重新构建输入
                    new_text = questions[i] + lib.datasets.SEP_TOKEN
                    new_text += lib.datasets.SEP_TOKEN.join(all_steps[i])
                    
                    # 重新编码
                    new_tokens = tokenizer.encode(new_text)[:x.shape[1]]
                    # 如果长度不足，填充
                    if len(new_tokens) < x.shape[1]:
                        padding = [lib.datasets.PAD_TOKEN_ID] * (x.shape[1] - len(new_tokens))
                        new_tokens.extend(padding)
                    
                    # 更新x
                    x[i] = torch.tensor(new_tokens, device=x.device)
                    
                    # 更新embedding
                    x_embed[i] = embedding_matrix[x[i]]
                    
                    # 更新z
                    z[i] = alpha_t[i] * x_embed[i] + sigma_t[i] * noise[i]
        
        # 计算平均SelfCheck损失
        if batch_size > reconst_bs:
            selfcheck_loss = selfcheck_loss / (batch_size - reconst_bs)
        else:
            selfcheck_loss = torch.tensor(0.0, device='cuda')
    
    # 设置bias scale
    bias_scale = min(1., (step + 1e-8) / (args.bias_warmup_steps + 1e-8)) if train_mode else 1.
    
    # 模型前向传播
    logits, x_reconst = modules['model'](
        z_selfcond=z, 
        gamma_selfcond=gamma, 
        embedding_matrix=embedding_matrix, 
        bias_scale=bias_scale,
        x_selfcond=x_selfcond,
        x_embed=x_embed if args.fix_src else None,
        src_mask=src_mask if args.fix_src else None
    )
    
    # 计算损失
    reconst_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        x.reshape(-1),
        reduction='none'
    ).reshape(x.shape)
    
    reconst_loss_masked = (reconst_loss * tgt_mask).sum(-1)
    reconst_loss = (reconst_loss_masked / tgt_mask.sum(-1)).mean()
    
    # 计算扩散损失
    eps_pred = noise - (z - alpha_t[:, None, None] * x_reconst) / sigma_t[:, None, None]
    diffusion_loss = (eps_pred ** 2).mean()
    
    # 计算总损失
    total_loss = args.reconst_weight * reconst_loss + args.diffusion_weight * diffusion_loss
    
    # 如果启用SelfCheck，加入SelfCheck损失
    if selfcheck_config and selfcheck_config.enabled:
        total_loss = total_loss + selfcheck_config.weight * selfcheck_loss
        return total_loss, reconst_loss, diffusion_loss, selfcheck_loss
    else:
        return total_loss, reconst_loss, diffusion_loss



def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('batch_size', 16)  # actual batch_size=batch_size*ngpus
    args.setdefault('dataset', 'gsm8k')
    args.setdefault('grad_accum_steps', 1) # 1
    args.setdefault('hook_freq', 500) # 100
    args.setdefault('lr', 3e-4) # 1.4e-3
    args.setdefault('lr_warmup_steps', 25) # 2500
    args.setdefault('bias_warmup_steps', 50) # 5000
    args.setdefault('lr_decay', True)
    args.setdefault('print_freq', 10) # 1000
    args.setdefault('save_weights', True)
    args.setdefault('steps', 9000) # 128*9k/400k~3ep
    args.setdefault('weights_path', None)
    args.setdefault('reconst_weight', 1.0)
    args.setdefault('dim', 2048)
    args.setdefault('n_blocks', 24)
    args.setdefault('n_heads', 32)
    args.setdefault('gamma_0', -3.)
    args.setdefault('gamma_1', 6.)
    args.setdefault('embed_dim', 16)
    args.setdefault('seq_len', 256)
    args.setdefault('weight_decay', 4e-5)
    args.setdefault('first_step', 0)
    args.setdefault('auto_resume', False)
    args.setdefault('decay_to_init', 0.)
    args.setdefault('ema', 0.)
    args.setdefault('beta1', 0.9)
    args.setdefault('beta2', 0.99)
    args.setdefault('selfcond', True)
    args.setdefault('clip_quantile', 0.95)
    args.setdefault('reconst_bs_ema', 0.997)
    args.setdefault('fix_src', False) # if True, src is not learned and is fixed in z as in DiffuSeq
    args.setdefault('cot', False) # cot=True refers to multi-pass diffusion, q+previous thoughts -> next thought
    args.setdefault('digit', True) # seperate each digit during tokenization
    args.setdefault('min_prob', 1.) # min prob in scheduled sampling
    args.setdefault('glance', False) # glance sampling in mp-dot


    args.setdefault('use_selfcheck', False)  # 是否启用SelfCheck
    args.setdefault('selfcheck_threshold', 0.7)  # SelfCheck置信度阈值
    args.setdefault('selfcheck_weight', 0.1)  # SelfCheck损失权重
    args.setdefault('selfcheck_apply_corrections', True)  # 是否应用SelfCheck修正
    



    set_args(args)
    lib.utils.print_args(args)

    set_seed(2024)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp32/bf16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

    (train_loader, valid_loader), (word2idx, idx2word), tokenizer = get_dataloaders(
        args.dataset, args.batch_size, args.seq_len, args.cot, args.digit, args.glance
    )
    train_iterator = infinite_loader(train_loader)

    test_loader = get_dataloader(args.dataset, 'test', 84, tokenizer, args.seq_len, args.cot)

    logging.info(f"world size: {lib.ddp.world_size()}")
    
    vocab_size = len(word2idx)
    logging.info(f'vocab_size: {vocab_size}')

    def create_modules(dim, n_heads):
        return {
            'noise_schedule': lib.models.NoiseSchedule().float(),
            'gamma_bounds': lib.models.GammaBounds(args.gamma_0, args.gamma_1).float(),
            'embedding_matrix': lib.models.EmbeddingMatrix(vocab_size, args.embed_dim).float(),
            'model': lib.models.DiffusionModel(dim, args.embed_dim, args.n_blocks, n_heads, vocab_size).float()
        }
    
    modules = create_modules(args.dim, args.n_heads)
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.cuda()
        logging.info(key+':')
        lib.utils.print_model(main)

    def load_weights(weights_path):
        logging.info(f'Loading weights from {weights_path}')
        for name, module in modules.items():
            module.load_state_dict(torch.load(
                os.path.join(weights_path, f'{name}.pt'),
                map_location=torch.device('cuda')
            ))

    if args.auto_resume:
        assert(args.save_weights)

    first_step = args.first_step
    if args.auto_resume and os.path.exists('model.pt'):
            load_weights('.')
            with open('step', 'r') as f:
                first_step = int(f.read()) + 1
    elif args.weights_path is not None:
        load_weights(args.weights_path)

    logging.info(f'Starting from step {first_step}')


    ddp_modules = {
        name: DDP(module, broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True
        )
        for name, module in modules.items()
    }

    logging.info('DDP initialized')

    emas = {
        name: lib.ema.EMA(module, args.ema)
        for name, module in modules.items()
    }

    decay_to_init = {
        name: lib.decay_to_init.DecayToInit(module, args.decay_to_init)
        for name, module in modules.items()
    }

    loss_ema_bias     = torch.tensor(1e-8).cuda()
    reconst_ema       = torch.tensor(1e-8).cuda()
    diffusion_ema     = torch.tensor(1e-8).cuda()
    reconst_sqr_ema   = torch.tensor(1e-8).cuda()
    diffusion_sqr_ema = torch.tensor(1e-8).cuda()
    reconst_bs_cache  = {}

    # infer_args for scheduled sampling and evaluation
    infer_args = args.copy()
    infer_args.update({'initial_noise_scale': 1.0,
                    'sampling_timesteps': 16, 
                    'score_temp': 0.5,
                    'dpm_solver': False,
                    'logit_sample': False,
                    'logit_temp': 0.5,
                    'runs': 1,
                    'apply_sc': False,
                    'cot_steps': 6,
                    'limit': False
    })
    infer_args = lib.utils.AttributeDict(infer_args)

    def forward(step=None, accum_step=None, accum_total=None, x_eval=None):
        """
        Train mode: step, accum_step (0~8), accum_total (8 gpus*1 grad_acc_steps)
        Eval mode: x_eval
        """
        nonlocal reconst_ema, diffusion_ema, reconst_sqr_ema, diffusion_sqr_ema

        train_mode = (x_eval is None)
        if train_mode:
            x, attn_mask, src_mask = next(train_iterator)
            x = x.cuda()
            attn_mask = attn_mask.cuda()
            src_mask = src_mask.cuda()
           
            batch_size = x.shape[0] * accum_total
            if step not in reconst_bs_cache:
                # Synchronize EMA vars
                reconst_ema       = lib.ddp.reduce_mean(reconst_ema)
                reconst_sqr_ema   = lib.ddp.reduce_mean(reconst_sqr_ema)
                diffusion_ema     = lib.ddp.reduce_mean(diffusion_ema)
                diffusion_sqr_ema = lib.ddp.reduce_mean(diffusion_sqr_ema)
                # Compute reconst_bs
                b = 1 / loss_ema_bias # Bias correction factor
                reconst_std   = (b*reconst_sqr_ema   - (b*reconst_ema)**2).clamp(min=0).sqrt()
                diffusion_std = (b*diffusion_sqr_ema - (b*diffusion_ema)**2).clamp(min=0).sqrt()
                reconst_bs = batch_size * (reconst_std / (1e-8 + reconst_std + diffusion_std))
                reconst_bs = int(reconst_bs.round().clamp(1, batch_size-1))
                reconst_bs_cache[step] = reconst_bs
            reconst_bs = reconst_bs_cache[step]
            avg_reconst_bs = float(reconst_bs)
        else:
            x, attn_mask, src_mask = x_eval
            x = x.cuda()
            attn_mask = attn_mask.cuda()
            src_mask = src_mask.cuda()
            batch_size = x.shape[0]
            reconst_bs = (batch_size // 8)  # no reconst loss if bs <=8
            reconst_bs += int(np.random.binomial(1, (batch_size % 8) / 8.))
            avg_reconst_bs = batch_size / 8.

        embedding_matrix = ddp_modules['embedding_matrix']()

        selfcond_mask = torch.zeros([batch_size], device='cuda')
        avg_selfcond_mask = 0.
        if args.selfcond:
            if train_mode:
                offset = int(np.random.randint(4))
                selfcond_mask[offset::4].add_(1)
                avg_selfcond_mask = 0.25
            else:
                selfcond_mask.add_(1)  # all perform selfcond in evaluation mode
                avg_selfcond_mask = 1.

        t = torch.empty([batch_size], device='cuda')
        # First entries of t are used for reconst_loss below
        t[:reconst_bs] = 0
        # Low-discrepancy sampler for the remaining entries of t
        t[reconst_bs:] = torch.arange(
            batch_size - reconst_bs, device='cuda')
        if train_mode:
            t[reconst_bs:] += float(np.random.RandomState(step).uniform())
        else:
            t[reconst_bs:] += float(np.random.uniform())
        t[reconst_bs:] /= batch_size - reconst_bs
        t.requires_grad = True

        if train_mode:
            batch_size //= accum_total
            selfcond_mask = selfcond_mask.chunk(accum_total)[accum_step]
            t = t.chunk(accum_total)[accum_step]
            reconst_bs = int(t.eq(0).sum())
            avg_reconst_bs /= accum_total

        selfcond_idx = selfcond_mask.nonzero()[:,0]

        with torch.enable_grad():
            # Don't propagate grads for the first reconst_bs entries of t
            gamma = torch.cat([
                ddp_modules['noise_schedule'](t[:reconst_bs]).detach(),
                ddp_modules['noise_schedule'](t[reconst_bs:])
            ])
            gamma_prime = autograd.grad(gamma.sum(), [t], create_graph=True)[0]
        # Edits gradients so that the noise schedule minimizes
        # E[loss^2] while the rest of the model minimizes E[loss].
        def set_grad_hook(tensor):
            if tensor.requires_grad:
                def grad_hook(grad):
                    handle.remove()
                    new_grad = torch.clone(grad.detach())
                    new_grad[reconst_bs:] *= 2. * (
                        grad_hook_loss[reconst_bs:].detach()
                    )
                    return new_grad
                handle = tensor.register_hook(grad_hook)

        gamma = gamma.clone()
        set_grad_hook(gamma)
        set_grad_hook(gamma_prime)
        gamma_0, gamma_1 = ddp_modules['gamma_bounds']()
        gamma = gamma_0 + (gamma_1 - gamma_0) * gamma
        gamma_prime = (gamma_1 - gamma_0) * gamma_prime

        gamma = torch.lerp(gamma, gamma.detach(), selfcond_mask)
        gamma_prime = torch.lerp(gamma_prime, gamma_prime.detach(), selfcond_mask)

        # Quantities derived from gamma, gamma_prime, gamma_1:
        alpha_squared = torch.sigmoid(-gamma)
        sigma_squared = torch.sigmoid(gamma)
        alpha = alpha_squared.sqrt()
        sigma = sigma_squared.sqrt()
        snr_prime = -(-gamma).exp() * gamma_prime # SNR = exp(-gamma)
        alpha_1 = torch.sigmoid(-gamma_1).sqrt()
        sigma_1 = torch.sigmoid(gamma_1).sqrt()
        
        x0_embed = None
        if train_mode and args.min_prob < 1:
            # Model forward pass for scheduled sampling
            with torch.no_grad():
                # get u for each t 
                timesteps_togo = (infer_args.sampling_timesteps*t).ceil()-1
                ratio = torch.rand(x.shape[0], device=x.device)
                gold_prob = sampling_gold_prob(step, args.steps, args.min_prob)
                use_pred = (ratio>gold_prob) & (timesteps_togo>0) 
                x0 = x.detach()
                if use_pred.any().item():
                    # print(step, gold_prob)
                    pred_x = generate_samples(x0[use_pred], src_mask[use_pred], modules, infer_args, timesteps_togo)
                    x0[use_pred] = pred_x
                    x0 = x0.detach()

            x0_embed = embedding_matrix[x0]
            x0_embed = torch.lerp(x0_embed, x0_embed.detach(), selfcond_mask.float()[:,None,None])

        x_embed = embedding_matrix[x]
        x_embed = torch.lerp(x_embed, x_embed.detach(), selfcond_mask.float()[:,None,None])
        
        z = torch.randn(
            [x.shape[0], x.shape[1], args.embed_dim],
            dtype=torch.float32, device='cuda'
        )
        z.mul_(sigma[:,None,None])
        z.add_(alpha[:,None,None] * (x_embed if x0_embed is None else x0_embed))

        if train_mode:
            bias_scale = min(1., (step + 1e-8) / (args.bias_warmup_steps + 1e-8))
        else:
            bias_scale = 1.

        # Model forward pass for self-conditioning
        x_selfcond = torch.zeros_like(z)
        if len(selfcond_idx) > 0:
            with torch.no_grad():
                z_selfcond = z[selfcond_idx]
                gamma_selfcond = gamma[selfcond_idx]
                attn_mask_selfcond = attn_mask[selfcond_idx]
                logits, x_reconst = ddp_modules['model'](
                    z_selfcond, gamma_selfcond, embedding_matrix, bias_scale, torch.zeros_like(z_selfcond),
                    x_embed=x_embed[selfcond_idx] if args.fix_src else None,
                    src_mask=src_mask[selfcond_idx] if args.fix_src else None
                )
                del logits

                x_selfcond[selfcond_idx] = x_reconst
                
        # Main model forward pass
        with torch.enable_grad():
            # model(z,t) -> x
            logits, x_reconst = ddp_modules['model'](
                z, gamma, embedding_matrix, bias_scale, x_selfcond,
                selfcond_mask=selfcond_mask,
                x_embed=x_embed if args.fix_src else None,
                src_mask=src_mask if args.fix_src else None
            )                
            
        loss_mask = torch.ones_like(src_mask).squeeze(-1) # bs, seq
        nll_loss_mask = attn_mask&(~src_mask).squeeze(-1)  # only calculate tgt loss without pad when checking nll loss

        loss_weight = torch.ones_like(src_mask, dtype=torch.float32).squeeze(-1) # bs, seq 

        # Loss terms
        if reconst_bs > 0:
            reconst_loss = lib.ops.cross_entropy(
                logits[:reconst_bs],
                x[:reconst_bs]
            )
            nll_reconst_loss = masked_loss(reconst_loss[:reconst_bs], 
                                       nll_loss_mask[:reconst_bs],
                                       loss_weight[:reconst_bs], dim=-1).double() # bs
            reconst_loss = masked_loss(reconst_loss[:reconst_bs], 
                                       loss_mask[:reconst_bs],
                                       loss_weight[:reconst_bs], dim=-1).double() # bs
        else:
            nll_reconst_loss = torch.tensor([0], device='cuda').double()
            reconst_loss = torch.tensor([0], device='cuda').double()
            
        alpha_1_masked = torch.lerp(alpha_1, alpha_1.detach(), selfcond_mask)[:,None,None]
        sigma_1_masked = torch.lerp(sigma_1, sigma_1.detach(), selfcond_mask)[:,None,None]
        prior_loss = lib.ops.gaussian_kl(
            (alpha_1_masked * x_embed),
            sigma_1_masked,
            torch.tensor(0., device='cuda'),
            torch.tensor(1., device='cuda')
        ).sum(dim=2)
        nll_prior_loss = masked_loss(prior_loss, nll_loss_mask, loss_weight)  # scalar
        prior_loss = masked_loss(prior_loss, loss_mask, loss_weight)  # scalar

        diffusion_loss = (x_embed - x_reconst).pow(2).sum(dim=-1).double() # bs,seq_len
        nll_diffusion_loss = masked_loss(diffusion_loss, nll_loss_mask, loss_weight, dim=-1) # bs
        nll_diffusion_loss = -0.5*(snr_prime * nll_diffusion_loss)
        diffusion_loss = masked_loss(diffusion_loss, loss_mask, loss_weight, dim=-1) # bs
        diffusion_loss = -0.5*(snr_prime * diffusion_loss)

        if train_mode:
            with torch.no_grad():
                loss_ema_bias.lerp_(     torch.tensor(1., device='cuda'),                                                   1 - args.reconst_bs_ema)
                reconst_ema.lerp_(       (args.reconst_weight * reconst_loss).sum()        / avg_reconst_bs,                1 - args.reconst_bs_ema)
                reconst_sqr_ema.lerp_(   (args.reconst_weight * reconst_loss).pow(2).sum() / avg_reconst_bs,                1 - args.reconst_bs_ema)
                diffusion_ema.lerp_(     diffusion_loss[reconst_bs:].sum()                 / (batch_size - avg_reconst_bs), 1 - args.reconst_bs_ema)
                diffusion_sqr_ema.lerp_( diffusion_loss[reconst_bs:].pow(2).sum()          / (batch_size - avg_reconst_bs), 1 - args.reconst_bs_ema)

        grad_hook_loss = diffusion_loss # Used above (weird variable scope)

        loss = (args.reconst_weight * reconst_loss).sum() / avg_reconst_bs
        loss += diffusion_loss[reconst_bs:].sum() / (batch_size - avg_reconst_bs)
        loss += prior_loss

        if args.selfcond:
            nll = (nll_reconst_loss * selfcond_mask[:reconst_bs]).sum() / (avg_reconst_bs * avg_selfcond_mask)
            nll += (nll_diffusion_loss[reconst_bs:] * selfcond_mask[reconst_bs:]).sum() / ((batch_size - avg_reconst_bs) * avg_selfcond_mask)
            nll += nll_prior_loss
        else:
            nll = nll_reconst_loss.sum() / avg_reconst_bs
            nll += nll_diffusion_loss[reconst_bs:].sum() / (batch_size - avg_reconst_bs)
            nll += nll_prior_loss

        return (
            loss,
            nll,
            reconst_loss.sum() / avg_reconst_bs,
            prior_loss,
            gamma_0,
            gamma_1,
            torch.tensor(reconst_bs).cuda(),
        )

    def choose_forward(step=None, accum_step=None, accum_total=None, x_eval=None, 
                    tokenizer=None, selfcheck_verifier=None, selfcheck_config=None):
        """
        根据是否启用SelfCheck选择前向传播函数
        """
        if selfcheck_config and selfcheck_config.enabled:
            return forward_with_selfcheck(step, accum_step, accum_total, x_eval, 
                                        tokenizer, selfcheck_verifier, selfcheck_config)
        else:
            return forward(step, accum_step, accum_total, x_eval)    
        

    learning_rates = {
        'model': args.lr,
        'noise_schedule': 1e-2,
        'gamma_bounds': 1e-2,
        'embedding_matrix': 1e-2,
    }

    weight_decays = {
        'model': args.weight_decay,
        'noise_schedule': 0.,
        'gamma_bounds': 1e-3,
        'embedding_matrix': 0.,
    }


    selfcheck_verifier = None
    selfcheck_config = None
    if args.use_selfcheck:
        # 确保lib/selfcheck.py已正确实现
        selfcheck_verifier = SelfCheckVerifier(tokenizer, modules, args)
        selfcheck_config = SelfCheckConfig(
            enabled=True,
            confidence_threshold=args.selfcheck_threshold,
            weight=args.selfcheck_weight,
            apply_corrections=args.selfcheck_apply_corrections
        )

    def optimizer_impl(param_groups, **kwargs):
        assert('weight_decay' not in kwargs)
        modules_seen = set()
        for i, param_group in enumerate(param_groups):
            weight_decay_set = False
            for name in modules:
                group_params = param_group['params']
                module_params = list(modules[name].parameters())
                if all([any([p is p2 for p2 in module_params]) for p in group_params]):
                    assert(not weight_decay_set)
                    assert(param_group['weight_decay'] == 0.)
                    param_group['weight_decay'] = (
                        weight_decays[name] / (param_group['lr']+1e-16)
                    )
                    weight_decay_set = True
                    modules_seen.add(name)
            assert(weight_decay_set)
        assert(all([name in modules_seen for name in modules]))

        return torch.distributed.optim.ZeroRedundancyOptimizer(param_groups,
            optimizer_class=optim.AdamW, parameters_as_bucket_view=True, **kwargs)

    param_groups = [
        {'params': modules[name].parameters(), 'lr': learning_rates[name]}
        for name in modules
    ]
    opt = mup.MuAdam(param_groups, impl=optimizer_impl, betas=(args.beta1, args.beta2))

    def compute_nll(data_iterator, seq_len=args.seq_len):
        with contextlib.ExitStack() as stack:
            for ema in emas.values():
                stack.enter_context(ema.enabled())
            stack.enter_context(torch.no_grad())
            total_nll = 0.
            n = 0
            for i, X in enumerate(data_iterator):
                nll = forward(x_eval=X)[1]
                total_nll += nll.item()
                n += 1
                # if i == steps:
                #     break
        return lib.ddp.reduce_mean(total_nll).item()/n

    all_val_nlls = []
    all_test_accs = []
    def hook(step):
        for decay in decay_to_init.values():
            decay.step(step, args.steps)

        for ema in emas.values():
            ema.step()

        if step % args.hook_freq == (args.hook_freq - 1):
            val_nll = compute_nll(iter(valid_loader))
            logging.info(f'NLL (val, seq_len={args.seq_len}): {val_nll}')
            all_val_nlls.append(val_nll)
            if args.seq_len != 256:
                val_nll_256 = compute_nll(iter(valid_loader), seq_len=256)
                logging.info(f'NLL (val, seq_len=256): {val_nll_256}')

            ## evaluate on test set
            acc = evaluate(infer_args, test_loader, tokenizer, modules, log_interval=False)
            all_test_accs.append(acc)

            if lib.ddp.rank() == 0:
                # Save weights
                if args.save_weights:
                    for name in modules:
                        with emas[name].enabled():
                            torch.save(modules[name].state_dict(), f'{args.save_weights_path}/{name}.pt')
                    with open(f'{args.save_weights_path}/step', 'w') as f:
                        f.write(str(step))
                    logging.info('Saved weights!')

                plt.clf()
                plt.plot(all_test_accs)
                plt.savefig(f'{args.save_weights_path}/test_acc.jpg')

                plt.clf()
                plt.plot(all_val_nlls)
                plt.savefig(f'{args.save_weights_path}/val_nll.jpg')
                
    logging.info('Starting train loop...')
    # 选择前向传播函数
    actual_forward = choose_forward if args.use_selfcheck else forward
    lib.utils.train_loop(
        actual_forward,
        opt,
        args.steps,
        names=['nll','reconst','prior','gamma_0','gamma_1','reconst_bs'],
        hook=hook,
        print_freq=args.print_freq,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay=args.lr_decay,
        amp_grad_scaler=False,
        grad_accum_steps=args.grad_accum_steps,
        ddp_models=ddp_modules.values(),
        first_step=first_step,
        clip_params=[
            param
            for module in modules.values()
            for param in module.parameters()
        ],
        clip_quantile=args.clip_quantile,
            # 添加SelfCheck相关参数
        tokenizer=tokenizer if args.use_selfcheck else None,
        selfcheck_verifier=selfcheck_verifier,
        selfcheck_config=selfcheck_config,
    )

    final_val_nll = compute_nll(iter(valid_loader))
    logging.info(f'Final val NLL: {final_val_nll}')
    if args.seq_len != 256:
        final_val_nll_256 = compute_nll(iter(valid_loader), seq_len=256)
        logging.info(f'Final val NLL (seq_len=256): {final_val_nll_256}')

    ## evaluate on test set
    test_args = infer_args.copy()
    test_args.update({'sampling_timesteps': 64, 'cot_steps': 12})
    test_args = lib.utils.AttributeDict(test_args)
    
    evaluate(test_args, test_loader, tokenizer, modules, log_interval=False)

    return all_val_nlls, final_val_nll

if __name__ == '__main__':
    fire.Fire(lib.ddp.wrap_main(main))
