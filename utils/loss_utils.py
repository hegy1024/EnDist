import torch

from .typing_utils import *

class LossInfos(object):
    draw_kwargs = Munch(
        colors=["b", "c", "g", 'r', "m"],
        markers=["o", "^", "*", "s", "p"],
        figsize=(20, 8),
        dpi=400,
        linestyle='-.',
        linewidth=5,
        markersize=15,
        titlesize=30,
        fontsize=20,
        labelsize=15,
        labelwidth=2,
        spine_linewidth=2,
        root_path='.img/loss'
    )

    def __init__(self, epochs: int, size: int, **kwargs):
        self.total_loss = DefaultDict(lambda: [0] * epochs)  # loss_name + epoch_loss
        self.len   = epochs
        self.size  = size  # number of train sample
        self.draw_kwargs.update(**kwargs)

    def clear(self):
        self.total_loss = DefaultDict(lambda: [0] * self.len)

    def update(self, epoch: int, id_loss: Union[Munch, EasyDict]):
        id_loss_items = [(u, float(v)) for u, v in id_loss.items()]
        # self.total_loss['total'][epoch] += sum([u for _, u in id_loss_items]) / self.size
        self.total_loss['total'][epoch] += sum([u for _, u in id_loss_items])
        for u, v in id_loss_items:
            it = self.total_loss.__getitem__(u)
            # it[epoch] += v / self.size
            it[epoch] += v

    def print(self, tgt: Optional[str] = None, epoch: int = -1, end: str = ' ' * 5):
        print(f"Epoch: {epoch:<4}", end=' ')
        if tgt is None:
            for u, v in self.total_loss.items():
                print(f"{u}: {v[epoch]:>10.6f}", end=' '*5)
            print(end)
        else:
            print(f"{tgt}: {self.total_loss.__getitem__(tgt)[epoch]:>10.6f}", end=' ' * 5)

    def draw(self, name_list: List[str], fig_name: str, save_fig: bool = False):
        fig = plt.figure(figsize=self.draw_kwargs.figsize, dpi=self.draw_kwargs.dpi)
        ax = fig.gca()
        lines = []
        x = list(range(1, self.len + 1))
        for i, name in enumerate(name_list):
            y = self.total_loss.get(name)
            line = ax.plot(
                x, y,
                color=self.draw_kwargs.colors[i],
                linestyle=self.draw_kwargs.linestyle,
                linewidth=self.draw_kwargs.linewidth,
                marker=self.draw_kwargs.markers[i],
                markersize=self.draw_kwargs.markersize,
                label=name
            )
            lines.append(line)
        ax.set_title(fig_name, fontsize=self.draw_kwargs.titlesize)
        ax.set_xlabel('Epoch', fontsize=self.draw_kwargs.fontsize)
        ax.set_ylabel('Loss', fontsize=self.draw_kwargs.fontsize)
        ax.tick_params(labelsize=self.draw_kwargs.labelsize, width=self.draw_kwargs.labelwidth)

        ax.spines['bottom'].set_linewidth(self.draw_kwargs.spine_linewidth)
        ax.spines['left'].set_linewidth(self.draw_kwargs.spine_linewidth)
        ax.spines['top'].set_linewidth(self.draw_kwargs.spine_linewidth)
        ax.spines['right'].set_linewidth(self.draw_kwargs.spine_linewidth)

        ax.legend(fontsize=self.draw_kwargs.fontsize)

        if not save_fig:
            plt.show()
        else:
            path = osp.join(self.draw_kwargs.root_path, f"{fig_name}.png")
            plt.savefig(path, dpi=self.draw_kwargs.dpi)
        plt.close()

def kl_div(logits: Tensor, standard: Union[str, Tensor] = 'uniform') -> Tensor:
    """
    计算给定分布和标准分布之间的kl散度
    Args:
        logits: 给定分布
        standard: 标准分布

    Returns: kl散度值
    """
    if standard == 'uniform':
        target = torch.ones_like(logits, device=logits.device)
    elif standard == 'gaussian':
        target = torch.randn_like(logits, device=logits.device)
    else:
        target = standard

    return F.kl_div(logits.log_softmax(dim=-1), target.log_softmax(dim=-1), reduction="batchmean", log_target=True)

def cross_entropy(src: Tensor, tgt: Tensor):
    return - ((src.softmax(dim=-1).log() * tgt.softmax(dim=-1)).sum(dim=-1)).sum()

def info_nce_loss(pos_samples: Tensor, neg_samples: Tensor, temp: float = 0.5):
    r"""
    计算info_nce损失。
    .. math::
        \begin{}
            \mathcal L_{NCE} = -\frac{1}{2N^2}\sum_{q\in \mathbb P\cup \mathbb N}\sum_{p\in\mathbb P}log\
            \frac{exp(\frac{h(q,p)}{\tau } )}{exp(\frac{h(q,p)}{\tau }) + \sum_{n\in\mathbb N}exp(\frac{h(q,n)}{\tau })}
        \end{}
    Args:
        pos_samples: 正样本集
        neg_samples: 负样本集
        sim_fn: 相似度计算函数
        temp:  温度系数
    """
    query_samples = torch.cat([pos_samples, neg_samples], dim=0)

    def info_nce(q: Tensor):
        pos_sim = F.cosine_similarity(q, pos_samples.unsqueeze(dim=1))
        neg_sim = F.cosine_similarity(q, neg_samples.unsqueeze(dim=1))
        return ((pos_sim / temp).exp().sum() /
                ((pos_sim / temp).exp().sum() + (neg_sim / temp).exp().sum())).log()

    loss = sum([info_nce(query) for query in query_samples])
    return - loss

def knn(tgt: Tensor, src: Tensor, k: int, mode: str = 'logits') -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if mode == 'embeds':
        bf_dist = torch.cdist(tgt, src)  # 求欧式距离
        bf_dist_rank = bf_dist.argsort()[:, :(k + 1)]

        k_nearist_neighbor = torch.full((1, src.size(0)), False).view(-1)
        k_nearist_neighbor[bf_dist_rank] = True

        return k_nearist_neighbor
    else:
        tgt_label    = tgt.argmax(dim=-1)
        pos_samples  = src.argmax(dim=-1) == tgt_label
        size         = min((~pos_samples).sum(), pos_samples.sum(), k)  # sample size
        bf_dist      = torch.cdist(tgt, src)                            # compute euler distance
        bf_dist_rank = bf_dist.argsort().view(-1)                       # sort and get index

        pos_sample   = torch.full((1, src.size(0)), False).view(-1)
        neg_sample   = torch.full((1, src.size(0)), False).view(-1)

        pos_sample[bf_dist_rank[:size]] = True

        return pos_sample, neg_sample


def nt_xent_loss(x1: Tensor, x2: Tensor, T: float = 0.5) -> Tensor:
    assert x1.dim() == 2 and x1.dim() == x2.dim()

    x1_abs, x2_abs = x1.norm(dim=1), x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = (sim_matrix / T).exp()

    pos_sim = sim_matrix.diag()
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - loss.log().mean()

    return loss

def triplet_loss(anchor: Tensor, pos_samples: Tensor, neg_samples: Tensor, margin: float=1.):
    loss_func = nn.TripletMarginLoss(margin=margin)
    return loss_func(anchor, pos_samples, neg_samples)