import builtins
import logging
import os
import random
import tempfile
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.distributed as dist
import torch.optim
import wandb

from target_fedspl import train_target_domain as train_target_fedspl
from utils_fedspl import configure_logger, NUM_CLASSES, use_wandb
import os



@hydra.main(version_base=None, config_path="configs", config_name="root")
def main(args):
    # enable adding attributes at runtime
    OmegaConf.set_struct(args, False)
    args.job_name = HydraConfig.get().job.name

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1

    ngpus_per_node = 1

    # Simply call main_worker function
    main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # seed each process
    temp_dir = tempfile.mkdtemp()
    os.environ['TMPDIR'] = temp_dir
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # if hasattr(torch, "set_deterministic"):
        #     torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Set process specific info
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    # Suppress printing if not master
    if args.distributed and args.gpu != 0:

        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.gpu)

        # Adjust Data Settings According to Multi-Processing
        args.data.batch_size = int(args.data.batch_size / args.ngpus_per_node)
        args.data.workers = int(
            (args.data.workers + args.ngpus_per_node - 1) / args.ngpus_per_node
        )

    work_dir = os.getcwd()
    os.makedirs(work_dir, exist_ok=True)
    args.log_dir = work_dir
    configure_logger(args.rank)
    logging.info(
        f"Dataset: {args.data.dataset},"
        + f" Target domains: {args.data.target_domains},"
        + f" Pipeline: {'fed' if args.train_fed else 'target'}"
    )

    ## Iterate over each domain
    args.data.image_root = os.path.join(args.data.data_root)
    args.model_fed.num_classes = NUM_CLASSES[args.data.dataset]
    if args.train_fed:
        for fed_domain in args.data.fed_domains:
            args.data.fed_domain = fed_domain
            if use_wandb(args):
                wandb.init(
                    project=args.project if args.project else args.data.dataset,
                    group=args.memo,
                    job_type=fed_domain,
                    name=f"seed_{args.seed}",
                    config=dict(args),
                )

            ## Main Loop

            if use_wandb(args):
                wandb.finish()
    else:
        for fed_domain in args.data.fed_domains:
            args.data.fed_domain = fed_domain
            print(fed_domain)
            for tgt_domain in args.data.target_domains:
                print(tgt_domain)
                if fed_domain == tgt_domain:
                    continue
                args.data.tgt_domain = tgt_domain

                if use_wandb(args):
                    wandb.init(
                        project=args.project if args.project else args.data.dataset,
                        group=args.memo,
                        job_type=f"{fed_domain}-{tgt_domain}-{args.sub_memo}",
                        name=f"seed_{args.seed}",
                        config=dict(args),
                    )

                # main loop
                if args.target_algorithm == "ours":
                    print(1)
                    train_target_fedspl(args)
                    
                if use_wandb(args):
                    wandb.finish()


if __name__ == "__main__":
    os.environ['HYDRA_FULL_ERROR'] = '1'
    wandb.login(key='22912696ba07dd45dfd8849bd017b4e2197e6914')
    
    main()
