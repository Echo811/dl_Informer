import numpy as np
import pandas as pd


def params_info(args):
    print('基本信息')
    print(f'model_name = {args.model}')
    print(f'data_path = {args.data_path}')
    print(f'data_X = {args.features}')
    print(f'data_Y = {args.target}')
    print(f'data_freq = {args.freq}')

    print('序列维度信息')
    print(f'data_seq_len = {args.seq_len}')
    print(f'data_label_len = {args.label_len}')
    print(f'data_pred_len = {args.pred_len}')

    print('en-de信息')
    print(f'enc_in = {args.enc_in}')
    print(f'dec_in = {args.dec_in}')
    print(f'c_out = {args.c_out}')
    print(f'd_model = {args.d_model}')
    print(f'n_heads = {args.n_heads}')

    print('其他基本信息')
    print(f'itr（试验次数） = {args.itr}')
    print(f'distil（蒸馏特征） = {args.distil}')
    print(f'train_epochs = {args.train_epochs}')
    print(f'batch_size = {args.batch_size}')

def read_result(path, shape_bool=True):
    # 从 'metrics.npy' 文件中读取度量值数组
    # metrics = [mae, mse, rmse, mape, mspe]
    metrics = np.load(path + 'metrics.npy')
    # 从 'pred.npy' 文件中读取预测结果数组
    preds = np.load(path + 'pred.npy')
    # 从 'true.npy' 文件中读取真实值数组
    trues = np.load(path + 'true.npy')
    # 打印读取的数据
    if shape_bool:
        print("Metrics:", metrics.shape)
        print("Predictions:", preds.shape)
        print("True Values:", trues.shape)
    else:
        print("Metrics\n", metrics)
        print("Predictions\n", preds)
        print("True Values\n", trues)
        print(preds - trues)

def show_data_basic_info(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    # 获取数据基本信息
    info = df.info()
    # 获取统计信息
    describe = df.describe()
    # 获取前几行数据
    head = df.head()
    # 输出信息
    print("Data Info:")
    print(info)
    print("\nData Description:")
    print(describe)
    print("\nFirst Few Rows of Data:")
    print(head)
