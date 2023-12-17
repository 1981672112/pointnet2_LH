"""
cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)

这段代码看起来是在一个分布式训练环境中获取分布式信息的操作。让我来逐步解释这段代码的含义：
    cfg.rank:
        这个变量很可能代表当前进程的分布式排名（rank）。
        在分布式训练中，每个参与训练的进程都有一个唯一的排名，用来标识进程在分布式环境中的位置。
        排名通常从0开始递增。
    cfg.world_size:
        这个变量很可能代表分布式训练中的总进程数量，也就是通常所说的分布式训练的“world size”。
    cfg.distributed:
        这个变量很可能是一个标志（Boolean值），用于表示当前代码是否在分布式训练模式下运行。
        如果为True，则意味着代码正在进行分布式训练，如果为False，则意味着代码在单机训练模式下运行。
    cfg.mp:
        这个变量可能代表分布式训练中所使用的多进程（multiprocessing）框架或库，
        可能是一种实现分布式训练的工具。
    dist_utils.get_dist_info(cfg):
        这是一个函数调用，很可能是用来获取分布式训练的相关信息的。
        该函数可能会接收一个配置（cfg）作为参数，并返回包含有关分布式训练的信息的数据结构。
        这些信息可能包括当前进程的排名、总进程数量、是否在分布式模式下等。

总之，这段代码的目的是获取在分布式训练环境中的相关信息，如当前进程的排名、总进程数量、是否在分布式模式下运行等，
并将这些信息分别赋值给cfg.rank、cfg.world_size、cfg.distributed和cfg.mp等变量，
以便在代码的其他部分使用这些信息来适应分布式训练环境。
具体的细节可能需要查看dist_utils.get_dist_info函数的实现来了解。
"""
import os

print(os.environ)
print(type(os.environ))  # <class 'os._Environ'>
print(len(os.environ))

"""
.get('MASTER_PORT', None): 这是一个字典方法调用。
get() 方法用于从字典中获取指定键（即环境变量名）的值。在这里，
它尝试获取名为 'MASTER_PORT' 的环境变量的值。如果该环境变量存在，则返回其值；如果不存在，则返回第二个参数（在这里是 None）作为默认值。


"""
os.environ["JOB_LOG_DIR"] = 'log_dir'
print(len(os.environ))
