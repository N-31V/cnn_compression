from typing import Dict
import torch


def profile(model, name, input_size, device='cuda'):
    model.to(device)
    model.eval()
    image = torch.randn(input_size, device=device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{name}'),
        with_flops=True
    ) as prof:
        outputs = model(image)
    return prof


def _flop(profiler) -> Dict:
    flops = {}
    for e in profiler.events():
        if e.flops != 0:
            if e.key not in flops.keys():
                flops[e.key] = {'flops': e.flops, 'n': 1}
            else:
                flops[e.key]['flops'] += e.flops
                flops[e.key]['n'] += 1
    return flops


def flop(model, name, input_size) -> Dict:
    prof = profile(model, name, input_size)
    flops = _flop(prof)
    return flops
