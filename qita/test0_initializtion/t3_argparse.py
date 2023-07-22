import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')
parser.add_argument('--epoch', dest='epochxx', type=int, default=10, help='train epoch number')
# 有dest 脚本中用args.epochxx 终端中用--epoch
args = parser.parse_args(args=['1', '2', '--sum', '--epoch=12'])  # args=['1', '2', '3']
print(args.accumulate(args.integers))
print(args.epochxx)
args.epochxx = 100
print(args.epochxx)

"""
(4) choices
用于界定参数的取值范围
(5) dest
使用parse_args()对参数进行解析后,一个属性对应一个参数的值,而该属性值正是dest的值，
默认情况下，对于位置参数，就是位置参数的值，
对于可选参数，则是去掉前缀后的值

(6) nargs
一般情况下，一个参数与一个操作参数(如:--version)关联，但nargs可以将多个参数与一个操作参数关联

N——一个正整数
'?'——一个或多个参数
(10) action
用来给ArgumenParser对象判断如何处理命令行参数，支持的操作如下：
"""
