import torch

"""
在Pytorch中，Tensor和tensor都用于生成新的张量
>>>  a = torch.Tensor([1, 2])
>>>  a
tensor([1., 2.])
>>> a=torch.tensor([1,2])
>>> a
tensor([1, 2])

"""
# 首先我们从根源上来看看torch.Tensor()和torch.tensor()区别。

# 1. torch.Tensor
# torch.Tensor()是Python类，更明确的说，是默认张量类型torch.FloatTensor()的别名，
# torch.Tensor([1,2]) 会调用Tensor类的构造函数__init__，生成单精度浮点类型的张量。
# >>> a=torch.Tensor([1,2])
# >>> a.type()
# 'torch.FloatTensor'


# 2. torch.tensor()
# torch.tensor()仅仅是Python的函数，函数原型是：torch.tensor(data, dtype=None, device=None, requires_grad=False)
# 其中data可以是：list, tuple, array, scalar等类型。
# torch.tensor()可以从data中的数据部分做拷贝（而不是直接引用），
# 根据原始数据类型生成相应的torch.LongTensor，torch.FloatTensor，torch.DoubleTensor。

# 三种情况
# >>> a = torch.tensor([1, 2])
# >>> a.type()
# 'torch.LongTensor'

# >>> a = torch.tensor([1., 2.])
# >>> a.type()
# 'torch.FloatTensor'

# >>> a = np.zeros(2, dtype=np.float64)
# >>> a = torch.tensor(a)
# >>> a.type()
# torch.DoubleTensor


x = torch.Tensor([[0.5000, 0.5003, 0.5011, 0.5003, 0.5000, 0.5006],
                  [0.5000, 0.5003, 0.5010, 0.5003, 0.5000, 0.5006],
                  [0.5000, 0.5003, 0.5011, 0.5003, 0.5000, 0.5006],
                  [0.5000, 0.5003, 0.5010, 0.5004, 0.5000, 0.5005],
                  [0.5000, 0.5003, 0.5010, 0.5003, 0.5000, 0.5006]])
y = torch.Tensor([2, 2, 4, 4, 5])

loss2 = torch.nn.CrossEntropyLoss()
# l = loss2(x, y) # RuntimeError: expected scalar type Long but found Float

l = loss2(x, y.long())
"""
有两种改法：
 
1、loss2(x,y.long())，损失函数里面要求，target的类型应该是long类型，input类型不做要求。
 
2、可以在创建target(也就是本文中的y),使用y=torch.tensor([2,2,4,4,5]),
   torch.tensor和torch.Tensor()，存在显著的区别
   
   具体区别可以参考：https://blog.csdn.net/weixin_42018112/article/details/91383574
 
"""
