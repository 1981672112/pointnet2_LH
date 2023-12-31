import copy


"""
可变数据类型：
    当该数据类型对应的变量的值发生了变化时，如果它对应的内存地址不发生改变，那么这个数据类型就是 可变数据类型。
不可变数据类型：
    当该数据类型对应的变量的值发生了变化时，如果它对应的内存地址发生了改变，那么这个数据类型就是 不可变数据类型。
总结：
    可变数据类型更改值后，内存地址不发生改变。
    不可变数据类型更改值后，内存地址发生改变。

"""
"""
https://blog.csdn.net/lovedingd/article/details/128257172?ops_
request_misc=%257B%2522request%255Fid%2522%253A%25221682321829168
00215049400%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%
255Fall.%2522%257D&request_id=168232182916800215049400&biz_id=0&utm
_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank
_ecpm_v1~rank_v31_ecpm-2-128257172-null-null.142^v86^insert_down28
,239^v2^insert_chatgpt&utm_term=%E6%B7%B1%E6%8B%B7%E8%B4%9D%E5%92%
8C%E6%B5%85%E6%8B%B7%E8%B4%9D%E7%9A%84%E5%8C%BA%E5%88%ABpython&spm=1018.2226.3001.4187
浅拷贝是对一个对象父级（外层）的拷贝，并不会拷贝子级（内部）。使用浅拷贝的时候，分为两种情况。
    第一种，如果最外层的数据类型是可变的，比如说列表，字典等，浅拷贝会开启新的地址空间去存放。
    第二种，如果最外层的数据类型是不可变的，比如元组，字符串等，浅拷贝对象的时候，还是引用对象的地址空间。


深拷贝和浅拷贝的区别：
    1.浅拷贝： 将原对象或原数组的引用直接赋给新对象，新数组，新对象／数组只是原对象的一个引用
    2.深拷贝： 创建一个新的对象和数组，将原对象的各项属性的“值”（数组的所有元素）拷贝过来，是“值”而不是“引用”
        不可变(immutable)对象类型
        int，float，decimal，complex，bool，str，tuple，range，frozenset，bytes
        可变(mutable)对象类型
        list，dict，set，bytearray，user-defined classes (unless specifically made immutable)
        
        简单说就是：
            浅拷贝只复制某个对象的引用，而不复制对象本身，新旧对象还是共享同一块内存。
            深拷贝会创造一个一模一样的对象，新对象和原对象不共享内存，修改新对象不会改变原对对象。
            在python中
                浅拷贝(copy())：拷贝父对象，不会拷贝对象内部的子对象。
                深拷贝(deepcopy())：是copy模块中的方法，完全拷贝了子对象和父对象


值传递和引用传递
    初步总结：不可变类型如int，str，tuple类型在传递参数时都是传值形式
            即函数内改变并不能影响函数外变量的值

    修改变量的值知识让它指向了一个新的对象，与原来对象的值没有关系，
    如果原来的值没有对象指向它，就会被python的GC回收
    
    可变类型如list，set，dict传递参数时是引用传递，函数内外多一个引用，引用计数器便+1，
    函数内外指向的都是同一块内存地址，因此，会改变原来对象的值

这与java的机制虽不同，但是最后调用的效果是一样的 如：
"""
"""
首先，广义上的复制包括以下四种情况：
    a=b;
    a.copy();
    copy.copy(a):
    copy.deepcopy(a):
"""
alist = [2, 5, [6, 'b']]

b = alist

c = alist.copy()
d = copy.copy(alist)

e = copy.deepcopy(alist)

alist[2].append(9)
alist.append('s')
"""
print(alist, b, c, d, e)  # [2, 5, [6, 'b', 9], 's'] [2, 5, [6, 'b', 9], 's'] [2, 5, [6, 'b', 9]] [2, 5, [6, 'b', 9]] [2, 5, [6, 'b']]
print(alist, 'alist')  # [2, 5, [6, 'b', 9], 's'] alist
print(b, 'b')  # [2, 5, [6, 'b', 9], 's'] b
print(c, 'c')  # [2, 5, [6, 'b', 9]] c
print(d, 'd')  # [2, 5, [6, 'b', 9]] d
print(e, 'e')  # [2, 5, [6, 'b']] e
"""

"""
可以看出：一种有三种情况:
    1 b=alist是赋值语句，alist变则b变，这是因为将alist赋值给b只是给alist指向的内存地址多加了一个引用b，
    结合上面引用传递的理解可知，此时b与alist指向同一块地址，一变则多变。
    
    2 c=alist.copy()与d=copy.copy(alist)都是浅拷贝，浅拷贝是对表层对象的拷贝，
    复杂情况如列表嵌套，对于列表中的子对象是拷贝一个引用，或者说加一个标签，因此子对象随着alist改变，而外层对象没有。
    c和d的区别在于，alist.copy()是内置方法，而copy.copy(alist)是copy模块中的函数，需要提前调用copy模块。
    
    3 d=copy.deepcopy(alist)是深拷贝，是主观意义上的复制，复制出来就是完全一个新的对象，跟以前再没有关系了。

这里结合拷贝值和添加引用来理解浅拷贝和深拷贝，
给可变对象赋值是给内存中的数据添加一个引用，不论赋值多少个，都是指向同一块内存，一变则多变；
而浅拷贝是拷贝表层对象的值，深层对象添加一个引用，
因此深层对象会随着原来的改变，而值拷贝依据传值一样，是不会改变的；
深拷贝就是语义化的复制了，对象跟以前没有任何关系。
"""

ls1 = [1, 2, 3, 4]
print(id(ls1))
print(type(ls1))
ls2 = [1, 2, 3, 4]
print(id(ls2))
print(type(ls2))
print(ls1 == ls2)  # True
print(ls1 is ls2)  # False
"is和==比较的对象的内容时不同的，"
"即：is比较的是两个对象的地址值，也就是说两个对象是否为同一个实例对象；"
"而==比较的是对象的值是否相等，其调用了对象的__eq__()方法。"
