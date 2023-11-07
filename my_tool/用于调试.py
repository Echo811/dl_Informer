import pickle


with open('./obj/train_data.txt', 'rb') as exp_f:
    # 使用pickle.load反序列化对象
    train_data = pickle.load(exp_f)
with open('./obj/testdata.txt', 'rb') as exp_f:
    # 使用pickle.load反序列化对象
    testdata = pickle.load(exp_f)
with open('./obj/predata.txt', 'rb') as exp_f:
    # 使用pickle.load反序列化对象
    predata = pickle.load(exp_f)

print(predata.__getitem__(0))

print(predata.__len__())
