import torch


def profile(model, name, input_size):
    model.to('cuda')
    model.eval()
    image = torch.randn(input_size, device='cuda')
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{name}'), with_flops=True) as prof:
            outputs = model(image)
    return prof


def _flop(profiler) -> int:
    flops = {}
    for e in profiler.events():
        if e.flops != 0:
            if e.key not in flops.keys():
                flops[e.key] = {'flops': e.flops, 'n': 1}
            else:
                flops[e.key]['flops'] += e.flops
                flops[e.key]['n'] += 1
    for k, v in flops.items():
        print(f"{k}: {v['flops']} flops, {v['n']} calls")
    return sum([v['flops'] for v in flops.values()])

def flop(model, name, input_size) -> int:
    prof = profile(model, name, input_size)
    return _flop(prof)
