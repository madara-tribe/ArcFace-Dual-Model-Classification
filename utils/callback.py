import os
from pathlib import Path
import torch

class CallBackModelCheckpoint(object):
    def __init__(self, config):
        Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = config.ckpt_dir

    def __call__(self, global_step, loss, backbone: torch.nn.Module, header: torch.nn.Module = None):
        torch.save(backbone.state_dict(), os.path.join(self.ckpt_dir, str(global_step)+'_'+str(loss)+'_'+"backbone.pth"))
        if header is not None:
            torch.save(header.state_dict(), os.path.join(self.ckpt_dir, str(global_step)+'_'+str(loss)+'_'+"header.pth"))



