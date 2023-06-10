import torch

class FlatModelState:
    def __init__(self, params_buf, grads_buf, grad_hooks, param_addrs, group_addrs={},
                 dist_rank=0, dist_world_size=1):
        assert params_buf.numel() == grads_buf.numel()
        assert params_buf.numel() % dist_world_size == 0
        self.params_buf = params_buf  # Flat buffer of parameters
        self.grads_buf = grads_buf  # Flat buffer of grads
        self.grad_hooks = grad_hooks  # Grad copy hooks (to keep in scope)
        self.param_addrs = param_addrs  # map param -> (start, end)
        self.group_addrs = group_addrs  # map group_key -> (start, end)
        ## self.end = max(x[1] for x in self.param_addrs.values())  # length of valid data in buffers
        self.dist_rank = dist_rank
        self.dist_world_size = dist_world_size


    def _range_of(self, group_key=None):
        if not group_key:
            return 0, self.params_buf.numel()
        else:
            return self.group_addrs[group_key]

    def _dist_range_of(self, group_key=None):
        slice_numel = self.params_buf.numel() // self.dist_world_size
        start = slice_numel * self.dist_rank
        end = start + slice_numel
        if group_key:
            group_start, group_end = self._range_of(group_key)
            start = max(start, group_start)
            end = min(end, group_end)
        return start, end

    def params(self, group_key=None):
        start, end = self._range_of(group_key)
        return self.params_buf[start:end]

    def grads(self, group_key=None):
        start, end = self._range_of(group_key)
        return self.grads_buf[start:end]
    
    def dist_params(self, group_key=None):
        start, end = self._dist_range_of(group_key)
        return self.params_buf[start:end]
    
    def dist_grads(self, group_key=None):
        start, end = self._dist_range_of(group_key)
        return self.grads_buf[start:end]

    @property
    def dtype(self):
        return self.params_buf.dtype

    @property
    def device(self):
        return self.grads_buf.dtype

def flatten_model(params, dist_rank=0, dist_world_size=1):
    assert len(params) > 0
    if isinstance(params, dict):
        all_params = [p for some_params in params.values() for p in some_params]
    else:
        all_params = params
    params_buf, param_addrs = flatten_params(all_params, dist_world_size)
    grads_buf = torch.zeros_like(params_buf)
    grad_hooks = {}  # TODO: maybe remove?
    group_addrs = {}
    if isinstance(params, dict):
        for group_name, group_params in params.items():
            group_start = min(param_addrs[p][0] for p in group_params)
            group_end = max(param_addrs[p][1] for p in group_params)
            group_addrs[group_name] = (group_start, group_end)
    return FlatModelState(params_buf, grads_buf, grad_hooks, param_addrs, group_addrs,
                          dist_rank, dist_world_size)


def round_up(x: int, alignment: int) -> int:
    residual = x % alignment
    if residual == 0:
        return x
    else:
        return x + alignment - residual

def flatten_params(params, dist_world_size=1):
    aparam = next(iter(params))
    # Align everything to 128 byte boundaries
    def _align(x):
        return round_up(x, 128 // aparam.element_size())
    param_addrs = {}
    start, end = 0, 0
    for p in params:
        end = start + p.numel()
        param_addrs[p] = (start, end)
        start = _align(end)

    buf_numel = round_up(end, dist_world_size)  # Ensure buf is divisible by world size for uniform all_gather
    params_buf = torch.empty(buf_numel, dtype=aparam.dtype, device=aparam.device)
    for p in params:
        start, end = param_addrs[p]
        with torch.no_grad():
            params_buf[start:end].view_as(p).copy_(p)
        p.data = params_buf[start:end].view_as(p.data)

    return params_buf, param_addrs

# def _make_grad_hook(param, grads_buf, start, end):
#     def hook(*args):
#         if param.grad is not None:
#             grads_buf[start:end].add_(param.grad.view(end - start))
#             param.grad = grads_buf[start:end].view_as(param)
#     return hook

# def flatten_grads_with_hooks(params, param_addrs):
#     aparam = next(iter(params))
#     grads_buf_size = max(x[1] for x in param_addrs.values())
#     grads_buf = torch.empty(grads_buf_size, dtype=aparam.dtype, device=aparam.device)
#     grad_hooks = {}
#     # for p in params:
#     if False:
#         start, end = param_addrs[p]
#         param_grad_acc = p.expand_as(p).grad_fn.next_functions[0][0]
#         grad_hook = _make_grad_hook(p, grads_buf, start, end)
#         param_grad_acc.register_hook(grad_hook)
#         grad_hooks[p] = param_grad_acc

#     return grads_buf, grad_hooks


# Next:
# - Simply try running this (don't even bother to zero grads) and try to match
# - Update to apex adamW with master weights -- we'll need to wrap this ourselves, I think
# - Do simple flat_master breakage - can start with one param_group and then add in multiple
# Also: look seriously at fully-sharded-data-parallel -- maybe we can use that in combination with our own master weights
# - It does look like we can get far with just FSDP on a casted model + apex Adam
# - if it turns out to suck, then we can do it ourselves
# - but that should allow a certain amount of focus on compile/graph
