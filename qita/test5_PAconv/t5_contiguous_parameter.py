# https://blog.csdn.net/HuanCaoO/article/details/104793667
# https://blog.csdn.net/kdongyi/article/details/108180250

import torch
import torch.nn as nn
import torch.nn.functional as F

tensor2 = torch.Tensor([123])
nn.Parameter(tensor2, requires_grad=True)
"""
1 nn.Parameter(data, requires_grad=True):
    torch.nn.Parameter()将一个不可训练的tensor转换成可以训练的类型parameter，
    并将这个parameter绑定到这个module里面。
    即在定义网络时这个tensor就是一个可以训练的参数了。
    使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。

2 nn.Parameter(tensor)和对一个tensor直接进行requires_grad=True的区别：
    nn.Parameter会直接把你的这个参数放到优化器里去优化
    requires_grad=True只是说这个参数有了梯度，你可以对他进行优化，但是它不会给你放到优化器里，你还是要自己去优化的。
"""

"""
使用contiguous() 
    如果想要断开这两个变量之间的依赖（x本身是contiguous的），
    就要使用contiguous()针对x进行变化，感觉上就是我们认为的深拷贝。
    当调用contiguous()时，会强制拷贝一份tensor，
    让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。


torch.contiguous()方法语义上是“连续的”，
经常与torch.permute()、torch.transpose()、torch.view()方法一起使用，
要理解这样使用的缘由，得从pytorch多维数组的低层存储开始说起：

touch.view()方法对张量改变“形状”其实并没有改变张量在内存中真正的形状，可以理解为：
    view方法没有拷贝新的张量，没有开辟新内存，与原张量共享内存；
    view方法只是重新定义了访问张量的规则，使得取出的张量按照我们希望的形状展现。
    
pytorch与numpy在存储MxN的数组时，均是按照行优先将数组拉伸至一维存储，比如对于一个二维张量
"""
t = torch.tensor([[2, 1, 3],
                  [4, 5, 9]])  # 23

'在内存中实际上是[2, 1, 3, 4, 5, 9]'
'按照行优先原则，数字在语义和在内存中都是连续的，'
'当我们使用torch.transpose()方法或者torch.permute()方法对张量翻转后，改变了张量的形状'

t2 = t.transpose(0, 1)
t3 = t2.contiguous()
# 可以看到t3与t2一样，都是3行2列的张量，此时再对t3使用view方法：
# ten2 = t2.view(-1)  # RuntimeError: view size is not compatible with input tensor's size and stride
ten3 = t3.view(-1)

"""
原因是：改变了形状的t2 语义上是3行2列的，在内存中还是跟t一样，没有改变，
导致如果按照语义的形状进行view拉伸，数字不连续，
此时torch.contiguous()方法就派上用场了
"""

print(t3.view(-1))
print(t.view(-1))

"""
可以看出contiguous方法改变了多维数组在内存中的存储顺序，以便配合view方法使用
torch.contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。
"""
