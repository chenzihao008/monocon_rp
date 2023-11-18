import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.monocon_engine import MonoconEngine,MonoconEngine_ddp
from utils.engine_utils import tprint, get_default_cfg, set_random_seed, generate_random_seed,export_cfg
from utils.decorators import decorator_timer

# Some Torch Settings
torch_version = int(torch.__version__.split('.')[1])
if torch_version >= 7:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False




def clean_up():
    dist.destroy_process_group()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8000'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def train_ddp(local_rank, arg) -> None:
    # Initialize init_process_group
    setup(local_rank,arg.gpus)
    # Initialize Engine
    self = MonoconEngine_ddp(arg.cfg,local_rank)
    assert torch.cuda.is_available(), "CUDA is not available."
    assert (self.epochs < self.target_epochs), \
        "Argument 'target_epochs' must be equal to or greater than 'epochs'."
   
    
    # Print Info
    self._print_engine_info()
        
    # Export Current Configuration
    export_cfg(self.cfg, os.path.join(self.root, 'config.yaml'))
    
    # # Resume Training if 'resume_from' is specified.
    # if (resume_from is not None):
    #     self.load_checkpoint(resume_from)
    #     tprint(f"Training resumes from '{resume_from}'. (Start Epoch: {self.epochs})")
    
    # Start Training
    tprint(f"Training will be proceeded from epoch {self.epochs} to epoch {self.target_epochs}.")
    tprint(f"Result files will be saved to '{self.root}'.")
    for epoch in range(self.epochs, self.target_epochs + 1):
        if self.sampler:
            self.sampler.set_epoch(epoch)
        print(f" Epoch {self.epochs:3d} / {self.target_epochs:3d} ".center(90, "="))
        
        avg_loss, elapsed_time = self.train_one_epoch()
        
        self.epoch_times.append(elapsed_time)
        time_info = self._get_time_info()
        
        print(f"\n- Average Loss: {avg_loss:.3f}")
        print(f"- Epoch Time: {time_info['epoch_time']}")
        print(f"- Remain Time: {time_info['remain_time']}")
        print(f"- Estimated End-Time: {time_info['end_time']}")
        
        # Validation
        if (self.val_period > 0) and (epoch % self.val_period == 0) and local_rank==0:
            self.model.eval()
            
            tprint(f"Evaluating on Epoch {epoch}...", indent=True)
            eval_dict = self.evaluate()

            # Write evaluation results to tensorboard.
            self._update_dict_to_writer(eval_dict, tag='eval')
            
            self.model.train()
            
            # Save Checkpoint (.pth)
            self.save_checkpoint(post_fix=None)
    
    # Save Final Checkpoint (.pth)
    if local_rank==0:
        self.save_checkpoint(post_fix='final')
    clean_up()


# Start Training from Scratch
# Output files will be saved to 'cfg.OUTPUT_DIR'.
if __name__=='__main__':
    # Get Config from 'config/monocon_configs.py'
    cfg = get_default_cfg()


    # Set Benchmark
    # If this is set to True, it may consume more memory. (Default: True)
    if cfg.get('USE_BENCHMARK', True):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        tprint(f"CuDNN Benchmark is enabled.")


    # Set Random Seed
    seed = cfg.get('SEED', -1)
    seed = generate_random_seed(seed)
    set_random_seed(seed)

    cfg.SEED = seed
    tprint(f"Using Random Seed {seed}")
    # engine.train()
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=torch.cuda.device_count(),
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--cfg', default=cfg, 
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = '192.168.3.6'              #
    os.environ['MASTER_PORT'] = '8000'                      #
    mp.spawn(train_ddp, nprocs=args.gpus, args=(args,))         #
    #########################################################
    # engine.run_DDP(1)
