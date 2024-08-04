from .typing_utils import *

def create_optimizer(model: Union[nn.Module, List[Tensor]], name: str = 'adam',
                     lr: float = 0.01, weight_decay: float = 0.0, **kwargs):
    del kwargs['grad_fn']
    params_info = Munch(params=model if not isinstance(model, nn.Module) else model.parameters(),
                        lr=lr, weight_decay=weight_decay, **kwargs)
    
    if name == 'adam':
        return torch.optim.Adam(**params_info)
    elif name == 'adamw':
        return torch.optim.AdamW(**params_info)
    elif name == "adadelta":
        return torch.optim.Adadelta(**params_info)
    elif name == "radam":
        return torch.optim.RAdam(**params_info)
    elif name == "sgd":
        params_info.momentum = 0.9
        return torch.optim.SGD(**params_info)
    else:
        raise ValueError("Invalid Optimizer")

def create_scheduler(optim: torch.optim.Optimizer, name: Optional[str] = None, **kwargs):
    if name == 'lambda':
        # 按照预先设定的lambda_fn进行衰减
        return torch.optim.lr_scheduler.LambdaLR(optim, kwargs['lambda_fn'])
    elif name == 'step':
        # 按照gamma进行等间隔衰减
        return torch.optim.lr_scheduler.StepLR(optim, kwargs['step_size'], kwargs['gamma'])
    elif name == 'exponetial':
        # 按照gamma进行每步衰减
        return torch.optim.lr_scheduler.ExponentialLR(optim, kwargs['gamma'])
    elif name == 'multistep':
        # 按照gamma进行非等间距衰减
        return torch.optim.lr_scheduler.MultiStepLR(optim, kwargs['milestones'], kwargs['gamma'])
    elif name == 'plateau':
        # 若最终指标停止改善，则降低学习率
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, **kwargs)
    elif name is None:
        return None
    else:
        return ValueError(f"lr_scheduler type {name} do not support now!")

def create_grad_fn(name: Optional[str] = None):
    if name is None:
        return lambda x: torch.stack(x).sum(dim=0)
    elif name == 'pc_grad':
        return pc_grad
    else:
        raise NotImplementedError

def get_grads(model: nn.Module) -> Tensor:
    r"""获取模型中参数的梯度"""
    grads = []
    for params in model.parameters():
        grads.append(params.grad.data.clone().flatten())
    return torch.cat(grads)

def set_grads(params_list: List[Tensor], new_grads: Tensor, mode: str = 'cover'):
    r"""更新待学习参数中的梯度"""
    start = 0
    for i, params in enumerate(params_list):
        dims = params.shape
        end  = start + dims.numel()
        if mode == 'cover':
            params.grad = new_grads[start: end].reshape(dims)
        elif mode == 'sum':
            if params.grad is None:
                params.grad  = new_grads[start: end].reshape(dims)
            else:
                params.grad += new_grads[start: end].reshape(dims)
        else:
            raise NotImplementedError(f'{mode} is not support now!!')
        start = end

def pc_grad(grad_list: List[Tensor]):
    r"""
    pc_grad: using grad inflection to modify conflicted grad
    """
    task_order = list(range(len(grad_list)))
    # Run tasks in random order
    random.shuffle(task_order)
    grad_pc = [g.clone() for g in grad_list]

    for i in task_order:
        for j in task_order:
            g1, g2 = grad_pc[i], grad_list[j]
            inner_product = min(torch.dot(g1, g2).item(), 0)  # if this number is larger than 0, don't modify the grad
            grad_pc[i]   -= inner_product / torch.dot(g2, g2) * g2
    # sum the grad for different domains
    new_grad = torch.stack(grad_pc).sum(dim=0)
    return new_grad

class Optimizer(object):
    def __init__(
        self,
        optim_configs: Union[Munch, EasyDict],
        scheduler_configs: Union[Munch, EasyDict],
        vars_list: List[Tensor]
    ):
        self.vars_list = vars_list

        self.optim     = create_optimizer(vars_list, **optim_configs)
        self.scheduler = create_scheduler(self.optim, **scheduler_configs)
        self.grad_fn   = create_grad_fn(optim_configs.grad_fn)

    def compute_gradients(self, loss_info, mode: str = 'cover'):
        r"""compute gradients for different domains and update gradients"""
        # loss_info = [loss for loss in loss_info if abs(loss) > 1e-10]

        domain_grads = [
            torch.cat([
                x.flatten() for x in torch.autograd.grad(
                    loss, self.vars_list, retain_graph=True, allow_unused=True)
            ], dim=0)
            for loss in loss_info
        ]
        new_grad = self.grad_fn(domain_grads)

        set_grads(self.vars_list, new_grad, mode)

    def step(self, *args):
        r"""update grad information"""
        self.optim.step()
        if self.scheduler is not None:
            # print("lr modify")
            self.scheduler.step(*args)

    def zero_grad(self):
        r"""clear gard"""
        self.optim.zero_grad()

    def __repr__(self):
        r"""update the name of optimizer"""
        name = self.optim.__class__.__name__
        if self.scheduler is not None:
            name += f" + {self.scheduler.__class__.__name__}"
        if self.grad_fn is not None:
            name += f" + {self.grad_fn.__name__}"
        return name