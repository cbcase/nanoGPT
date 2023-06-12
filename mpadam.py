import torch
from amp_C import multi_tensor_adam_capturable_master
from apex.multi_tensor_apply import multi_tensor_applier


class MixedPrecisionAdamW:
    def __init__(
        self,
        params_grads_options,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    ):
        self.dummy_overflow = torch.cuda.IntTensor([0])
        self.float_one = torch.cuda.FloatTensor([1.0])
        self.int_one = torch.cuda.IntTensor([1])
        self.param_groups = []
        for param, grads, options in params_grads_options:
            m, v, w32 = [torch.zeros_like(param, dtype=torch.float32) for _ in range(3)]
            step = torch.zeros(1, dtype=torch.int, device=param.device)
            with torch.no_grad():
                w32.copy_(param)
            group = {
                "params": param,
                "grads": grads,
                "m": m,
                "v": v,
                "step": step,
                "w32": w32,
            } | options
            if "lr" not in group:
                # TODO: fix "lr on device" handling here
                group["lr"] = torch.tensor(lr, dtype=torch.float32, device=param.device)
            self.param_groups.append(group)
        self.default_betas = betas
        self.default_eps = eps
        self.default_weight_decay = weight_decay
        self.use_apex = True

    def step(self):
        for group in self.param_groups:
            step = group["step"]
            step += self.int_one
            beta1, beta2 = group.get("betas", self.default_betas)
            if self.use_apex:
                multi_tensor_applier(
                    multi_tensor_adam_capturable_master,
                    self.dummy_overflow,
                    [
                        [group["grads"]],
                        [group["params"]],
                        [group["m"]],
                        [group["v"]],
                        [group["w32"]],
                    ],
                    group["lr"],
                    beta1,
                    beta2,
                    group.get("eps", self.default_eps),
                    step,
                    1,  # adam_w_mode = True
                    1,  # bias_correction = True
                    group.get("weight_decay", self.default_weight_decay),
                    self.float_one,
                )
            else:
                grads = group["grads"]
                params = group["params"]
                m = group["m"]
                v = group["v"]
                w32 = group["w32"]
                lr = group["lr"].item()
                decay = group.get("weight_decay", self.default_weight_decay)
                eps = group.get("eps", self.default_eps)
                stepval = step.item()
                w32 -= (lr * decay) * w32
                m = beta1 * m + (1 - beta1) * grads
                v = beta2 * v + (1 - beta2) * (grads * grads)
                mhat = m / (1 - beta1**stepval)
                vhat = v / (1 - beta2**stepval)
                w32 -= lr * mhat / (torch.sqrt(vhat) + eps)
                group["w32"] = w32
                group["m"] = m
                group["v"] = v
                params.copy_(w32)

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            group["grads"].zero_()
