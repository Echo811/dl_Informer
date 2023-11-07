import pickle

import sys, os
sys.path.append('..')

def save_(obj_dic, path='my_tool/obj01/'):
    print(os.getcwd())
    for obj_name in obj_dic:
        obj = obj_dic[obj_name]
        f = open(path+f'{obj_name}.txt', 'wb')
        pickle.dump(obj=obj, file=f)
        print(f'-----------保存{obj_name}对象成功！------------')
