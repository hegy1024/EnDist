import json
import logging
import functools

from termcolor import colored

from utils.typing_utils import *
from explain.explanation import Explanation

class _ColorfulFormatter(logging.Formatter):
    """设置输出内容的颜色"""
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(name, save_dir, distributed_rank=0, filename="log.txt", color=True, abbrev_name=None):
    """
    Initialize logger.
    """
    check_dir(save_dir, "Log ")
    print(f"Result Save at: {osp.join(save_dir, filename)}")
    logger = logging.getLogger(name)  # 实例化日志记录器
    logger.setLevel(logging.DEBUG)  # 设置日志等级
    logger.propagate = False  # 每个日志记录器独立配置，不再使用上一级日志器的配置

    if abbrev_name is None:
        abbrev_name = "ugait" if name == "ugait" else name

    if distributed_rank > 0:
        # 分布式训练时使用，只对主进程记录日志
        # 由于本项目几乎不使用分布式训练，这里默认设置为0
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)  # 创建流处理程序
    ch.setLevel(logging.DEBUG)  # 设置日志等级
    if color:
        # 创建格式处理程序
        # 设置日志内容的输出样式
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=str(abbrev_name),
        )
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
        )
    ch.setFormatter(formatter)  # 将格式处理程序应用到流处理程序
    logger.addHandler(ch)  # 将流处理程序应用到日志器

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def check_dir(dir_path: str, other_infos: str = ''):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)
    # print(f"{other_infos}Save Path: {dir_path:->30}")


def print_dict(dic: Union[dict, Munch, EasyDict], name: Optional[Dict] = None, end: str = '', shift: bool = False):
    if name is not None:
        for u, v in name.items():
            print(f"{u}: {v:>8}", end=' ')

    for u, v in dic.items():
        if isinstance(v, float):
            v = round(v, 6)
        print(f"{u: >3}:{v:>10}", end=end)
    if shift: print()


def save_arguments(cfgs: Munch):
    save_path = osp.join(cfgs.config_path, f"{cfgs.explainer.value}")
    check_dir(save_path)

    infos = cfgs.copy()
    infos.pop("explainer")
    infos.pop("data")
    infos.pop("explain_type")

    for u in cfgs.keys():
        if 'path' in u: infos.pop(u)

    file_name = osp.join(save_path, f"{cfgs.data.value}.json")
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(infos, f, ensure_ascii=False, indent=4)
    print(f"Configs saved at {file_name}")

def read_arguments(cfgs: Munch, save_path: Optional[str] = None):
    if save_path is None:
        save_path = osp.join(cfgs.config_path, f"{cfgs.explainer.value}/{cfgs.data.value}.json")

    assert osp.exists(save_path)
    with open(save_path, 'r', encoding='utf-8') as f:
        infos = Munch.fromDict(json.load(f))
    cfgs.update(infos)

    return cfgs

def save_explanations(explanations: List[Explanation],
                      explainer : str,
                      root      : str,
                      dataset   : str,
                      save_name : str = "best_explanation"):
    r"""
    Save explanations result to file.
    """
    save_path  = osp.join(root, dataset, f"{explainer}")
    check_dir(save_path, 'Explanations Result ')
    file_path   = osp.join(save_path, f"{save_name}.pt")
    torch.save(explanations, file_path)

    print(f"Explanation Save at {file_path}")

def load_explanation(explainer: str,
                     root     : str,
                     dataset  : str,
                     save_name: str = "best_explanation"):
    r"""
    Load explanations result to file.
    """
    save_path  = osp.join(root, dataset, f"{explainer}")
    # check_dir(save_path, 'Explanations Result ')
    file_path   = osp.join(save_path, f"{save_name}.pt")
    print(f"load_explanation at {file_path}")
    return torch.load(file_path, map_location="cuda:2")
