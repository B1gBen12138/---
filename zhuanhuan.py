# -*-coding=utf8-*-
from xpinyin import Pinyin
import os
import pypinyin
import sys
import re
resume_rootdir = '.'

mo=r'[\u4e00-\u9fa5]'
s='苹果-开灯1353.jpg'
s1=list(s)
#s1[0]='s'
res=re.findall(mo,s)
#print(res)
for i in res:
    #print(i)
    result = pypinyin.pinyin(i, style=pypinyin.NORMAL)
    #print(result[0][0])
    #print(result)
    #print(i)
    for j in range(len(s1)):
        #print(i)
        #print(s[j])
        if i==s1[j]:
            print('pipei')
            print('检测:'+i)
            print('原字符串：'+s[j])
            #s.replace(i,result[0][0][0])
            #s[j]==result[0][0]
            s1[j]=result[0][0][0]
            print(result[0][0][0])
    #s.replace(str(res[i]),'')
    #s[i]=
list3=''.join(s1)
print(list3)

def rename():
    print(u'重命名开始！')
    pin = Pinyin()
    llist = os.listdir('E:/test')
    result = pypinyin.pinyin('苹果',style=pypinyin.NORMAL)
    print(result)
    '''
    for i in range(0, len(llist)):
        print(u'现在进行第{}个'.format(i))
        resume = os.path.join(resume_rootdir, llist[i])
        print(resume)
        if os.path.isfile(resume):
            print('haha')
            obj = os.path.basename(resume)
            print(obj)
            if obj[0] == '.':
                continue

            print(u'开始处理  {}'.format(obj))
            pinyin_name = pin.get_pinyin(obj.decode('utf-8'), "")
            print(u'{} 新名字是:{}'.format(obj, pinyin_name))
            Newdir = os.path.join(resume_rootdir, pinyin_name);  # 新的文件路径
            os.rename(resume, Newdir)  # 重命名
    print(u'重命名结束！')


rename()
'''
