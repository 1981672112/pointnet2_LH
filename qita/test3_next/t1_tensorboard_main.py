"""
1 tensorboard画图
tensorboard --logdir=tswenjian
第一个参数可简单理解为保存到 tensorboard 日志文件中的标量图像的名称
第二个参数可简单理解为图像的 y 轴数据
第三个参数可简单理解为图像的 x 轴数据
第四个参数都是可选参数，用于记录发生的时间，默认为 time.time()

"""

from torch.utils.tensorboard import SummaryWriter
import numpy as np

np.random.seed(20200910)
writer = SummaryWriter('tswenjian')  # 保存位置
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

writer.close()

"""
2 判断字符是否在字符串中
"""
s = 'dp'
strs = 'dp_fj'
if s in strs:
    print('是的')

"""
3
这段代码是一个基于PyTorch实现的点云分割算法，其主要功能是进行点云数据的训练和验证。该算法使用了许多PyTorch的库和工具，
如argparse、torch、torch.nn、torch.utils.tensorboard、torch_scatter、torch.nn.functional等。
在训练过程中，该算法使用了各种数据增强技术，如旋转、平移、缩放、加噪声等。
同时，该算法使用了多种损失函数和优化器进行模型训练，如交叉熵损失、Dice损失、Adam优化器、SGD优化器等。
在验证过程中，该算法使用了混淆矩阵和多个指标进行评估，如平均部分交并比、平均实例交并比等。
此外，该算法还使用了Wandb库进行结果可视化和记录。最后，该算法支持分布式训练，可以在多个GPU上同时运行。

if main：
    具体来说，这段代码的主要功能包括：
    使用argparse库解析命令行参数，其中必须指定一个配置文件的路径（args.cfg）。
    使用EasyConfig对象读取指定的配置文件，并将其保存到cfg对象中。如果命令行参数中有一些选项（opts），那么也会将这些选项更新到cfg对象中。
    
    如果配置文件中没有指定随机种子（cfg.seed），则随机生成一个整数作为种子。
    使用dist_utils.get_dist_info(cfg)函数获取当前程序的分布式训练相关信息，
    包括当前进程的编号（cfg.rank）、总进程数（cfg.world_size）和是否使用分布式训练（cfg.distributed），并将这些信息保存到cfg对象中。
    
    根据配置文件和命令行参数生成一个实验目录（cfg.run_dir）和日志目录（cfg.log_dir），并将这些信息保存到cfg对象中。
    同时，将一些标记（tags）添加到cfg对象中，例如任务名称、模式、配置文件名称、使用的GPU数量和随机种子等。
    如果当前是从上一个训练中断后恢复训练（cfg.mode == 'resume'），则会加载之前训练的模型参数并继续训练。
    否则，会根据实验目录创建一个新的实验，并将一些额外的标记添加到cfg对象中，例如端口号等。
    将cfg对象保存到一个YAML文件中，并将运行的配置文件拷贝到实验目录中。
    如果使用多进程训练（cfg.mp == True），则会使用mp.spawn函数启动多个进程来进行分布式训练。否则，会使用main函数在当前进程中进行单进程训练。
"""

"""
这段代码使用 Python 的 argparse 模块来解析命令行参数。
首先，创建了一个 ArgumentParser 对象，并指定了程序的描述信息 'ShapeNetPart Part segmentation training'。
然后，使用 add_argument() 方法添加一个名为 '--cfg' 的参数，指定了该参数的类型是字符串（type=str），必须要提供该参数的值（required=True），
并给出了该参数的帮助信息。最后，调用 parse_known_args() 方法来解析命令行参数，返回一个元组，
其中第一个元素 args 是一个包含了解析后的命令行参数的 Namespace 对象，
第二个元素 opts 是一个列表，包含了所有无法解析的参数。
"""

"""
这行代码创建了一个 EasyConfig 对象，这是一个自定义的配置类，
用于加载和保存训练/测试过程中的各种参数和设置。
在下一行，配置文件的路径会被读取并传递给 load() 函数，以便将其中的参数和设置加载到 cfg 对象中。
"""

"""
这行代码将yaml配置文件中的所有键值对加载到EasyConfig对象中。 
args.cfg 是传递给程序的配置文件路径，
recursive=True 表示递归地加载所有父级default YAML文件。
加载后，EasyConfig对象中将包含从配置文件中读取的所有参数键值对。
"""

"""
cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)

cfg.rank：当前进程的rank，用于区分不同的进程。如果不是分布式训练，cfg.rank 的值默认为0。
cfg.world_size：总进程数，用于指定分布式训练时的进程数。
cfg.distributed：是否进行分布式训练，如果是，该值为True。
cfg.mp：表示使用的分布式训练框架，比如PyTorch的torch.distributed或Horovod等。

这里调用了 dist_utils.get_dist_info(cfg) 函数，该函数通过解析配置文件中的相关配置信息来确定当前是否进行分布式训练，
如果是，则根据选用的分布式训练框架来获取 rank 和 world_size。
"""

"""
yaml.dump(cfg, f, indent=2) 
这行代码会将 Python 字典类型的 cfg 对象转换为 YAML 格式，并写入文件流 f 中。indent=2 参数指定缩进空格数为 2。
"""

"""
model:

BasePartSeg(
  (encoder): PointNextEncoder(
                                (encoder): Sequential(
                                                      (0): Sequential(
                                                                      (0): SetAbstraction(
                                                                                          (convs): Sequential(
                                                                                                              (0): Sequential(
                                                                                                                              (0): Conv1d(7, 32, kernel_size=(1,), stride=(1,)
                                                                                                                              )
                                                                                                               )
                                                                                           )
                                                                       )
                                                        )
                                                      (1): Sequential(
                                                                     (0): SetAbstraction(
                                                                                         (skipconv): Sequential(
                                                                                                                (0): Conv1d(32, 64, kernel_size=(1,), stride=(1,)
                                                                                                                )
                                                                                          )
                                                                     (act): ReLU(inplace=True)
                                                                     (convs): Sequential(
                                                                                         (0): Sequential(
                                                                                                         (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                         (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                         (2): ReLU(inplace=True)
                                                                                                         )
                                                                                         (1): Sequential(
                                                                                                         (0): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                         (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                         (2): ReLU(inplace=True)
                                                                                                        )
                                                                                         (2): Sequential(
                                                                                                         (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                         (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                       )
                                                                                          )
                                                                     (grouper): QueryAndGroup()
                                                                                          )
                                                                      )
                                                      (2): Sequential(
                                                                     (0): SetAbstraction(
                                                                                        (skipconv): Sequential(
                                                                                                                (0): Conv1d(64, 128, kernel_size=(1,), stride=(1,)
                                                                                                                )
                                                                                         )           
                                                                     (act): ReLU(inplace=True)
                                                                     (convs): Sequential(
                                                                                        (0): Sequential(
                                                                                                          (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                          (2): ReLU(inplace=True)
                                                                                                        )
                                                                                        (1): Sequential(
                                                                                                          (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                          (2): ReLU(inplace=True)
                                                                                                       )
                                                                                        (2): Sequential(
                                                                                                          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                       )
                                                                                         )
                                                                     (grouper): QueryAndGroup()
                                                                                    )
                                                                      )
                                                      (3): Sequential(
                                                                     (0): SetAbstraction(
                                                                                        (skipconv): Sequential(
                                                                                        (0): Conv1d(128, 256, kernel_size=(1,), stride=(1,)
                                                                                                           )
                                                                                    )
                                                                     (act): ReLU(inplace=True)
                                                                     (convs): Sequential(
                                                                                    (0): Sequential(
                                                                                                      (0): Conv2d(131, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                      (2): ReLU(inplace=True)
                                                                                                    )
                                                                                    (1): Sequential(
                                                                                                      (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                      (2): ReLU(inplace=True)
                                                                                                    )
                                                                                    (2): Sequential(
                                                                                                      (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                    )
                                                                                     )
                                                                     (grouper): QueryAndGroup()
                                                                                        )       
                                                                      )
                                                      (4): Sequential(
                                                                      (0): SetAbstraction(
                                                                                            (skipconv): Sequential(
                                                                                                                    (0): Conv1d(256, 512, kernel_size=(1,), stride=(1,)
                                                                                                                    )
                                                                                            )
                                                                      (act): ReLU(inplace=True)
                                                                      (convs): Sequential(
                                                                                            (0): Sequential(
                                                                                                              (0): Conv2d(259, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                              (2): ReLU(inplace=True)
                                                                                                            )
                                                                                            (1): Sequential(
                                                                                                              (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                              (2): ReLU(inplace=True)
                                                                                                            )
                                                                                            (2): Sequential(
                                                                                                              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                                                                                                              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                             )
                                                                                             )
                                                                      (grouper): QueryAndGroup()
                                                                                            )
                                                                       )
                                                      )
                                 )
  (decoder): PointNextPartDecoder(
                                  (global_conv2): Sequential(
                                                              (0): Sequential(
                                                                              (0): Conv1d(512, 128, kernel_size=(1,), stride=(1,)
                                                                              )
                                                              (1): ReLU(inplace=True)
                                                               )
                                  )
                                  (global_conv1): Sequential(
                                                              (0): Sequential(
                                                                              (0): Conv1d(256, 64, kernel_size=(1,), stride=(1,)
                                                                                )
                                                               (1): ReLU(inplace=True)
                                                                 )
                                    )
    (decoder): Sequential(
                          (0): Sequential(
                                        (0): FeaturePropogation(
                                                                  (convs): Sequential(
                                                                                        (0): Sequential(
                                                                                                          (0): Conv1d(304, 32, kernel_size=(1,), stride=(1,), bias=False)
                                                                                                          (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                          (2): ReLU(inplace=True)
                                                                                                         )
                                                                                        (1): Sequential(
                                                                                                          (0): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
                                                                                                          (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                          (2): ReLU(inplace=True)
                                                                                                          )
                                                                                         )
                                                                   )
                                         )
                          (1): Sequential(
                                        (0): FeaturePropogation(
                                                                (convs): Sequential(
                                                                                        (0): Sequential(
                                                                                                          (0): Conv1d(192, 64, kernel_size=(1,), stride=(1,), bias=False)
                                                                                                          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                          (2): ReLU(inplace=True)
                                                                                                        )
                                                                                        (1): Sequential(
                                                                                                          (0): Conv1d(64, 64, kernel_size=(1,), stride=(1,), bias=False)
                                                                                                          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                          (2): ReLU(inplace=True)
                                                                                                        )
                                                                                        )
                                                                 )
                                          )
                          (2): Sequential(
                                        (0): FeaturePropogation(
                                                                (convs): Sequential(
                                                                                    (0): Sequential(
                                                                                                      (0): Conv1d(384, 128, kernel_size=(1,), stride=(1,), bias=False)
                                                                                                      (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                      (2): ReLU(inplace=True)
                                                                                                    )
                                                                                    (1): Sequential(
                                                                                                      (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
                                                                                                      (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                      (2): ReLU(inplace=True)
                                                                                                    )
                                                                                      )
                                                                 )
                                         )
                          (3): Sequential(
                                        (0): FeaturePropogation(
                                                               (convs): Sequential(
                                                                                   (0): Sequential(
                                                                                                   (0): Conv1d(768, 256, kernel_size=(1,), stride=(1,), bias=False)
                                                                                                   (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                   (2): ReLU(inplace=True)
                                                                                                   )
                                                                                   (1): Sequential(
                                                                                                  (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
                                                                                                  (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                                                                  (2): ReLU(inplace=True)
                                                                                                     )
                                                                                     )
                                                                    )
                                         )
                          )
                 )
  
  (head): SegHead(
                  (head): Sequential(
                                      (0): Sequential(
                                                        (0): Conv1d(96, 96, kernel_size=(1,), stride=(1,), bias=False)
                                                        (1): BatchNorm1d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                                        (2): ReLU(inplace=True)
                                                        )
                                      (1): Dropout(p=0.5, inplace=False)
                                      (2): Sequential(
                                                        (0): Conv1d(96, 50, kernel_size=(1,), stride=(1,))
                                                      )
                                       )
                   )
)
"""

"""
with np.printoptions(precision=2, suppress=True): 
"with np.printoptions(precision=2, suppress=True):" 这行代码设置了 numpy 数组的打印选项。
具体而言，它会将打印精度设置为小数点后两位，同时禁止使用科学计数法打印数组。这样可以使数组的打印结果更加易读。
"""

"""
lr = optimizer.param_groups[0]['lr']
在 PyTorch 中，优化器(optimizer)是一个用于更新模型参数的工具，它可以在训练过程中自动地计算梯度并更新参数。
同时，优化器还会管理一些超参数(hyperparameters)，例如学习率(learning rate)、权重衰减(weight decay)等，以控制模型训练的速度和效果。
在这行代码中，我们使用了 PyTorch 中的优化器(optimizer)对象，通过索引0来获取优化器的第一个参数组(param_groups[0])，
然后再从这个参数组中获取学习率(lr)。具体而言，param_groups 是一个列表(list)对象，其中每个元素都是一个字典(dict)对象，
用于存储参数组的超参数和其他相关信息。
在默认情况下，优化器只有一个参数组，因此我们可以通过索引0来获取它。而在这个参数组中，我们可以通过 'lr' 这个键(key)来获取学习率的值。
因此，这行代码的作用是获取 PyTorch 优化器中第一个参数组的学习率，并将其赋值给变量 lr，以便在模型训练过程中使用。
"""

"""
dist.destroy_process_group() 是 PyTorch 中的一个函数，用于销毁分布式训练过程中的进程组。
在分布式训练中，多个进程之间需要协同工作来完成模型的训练。进程组是一组协同工作的进程，可以通过指定一个唯一的名称来创建进程组。
一旦进程组完成任务或不再需要，就需要销毁进程组来释放资源并终止进程。

dist.destroy_process_group() 函数接受一个参数，即要销毁的进程组的句柄。
这个句柄是在创建进程组时分配的，用于唯一标识进程组。
调用 dist.destroy_process_group() 函数会终止进程组中的所有进程，并释放与该进程组相关的所有资源。
"""
