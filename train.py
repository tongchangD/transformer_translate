import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import dill as pickle


def train_model(model, opt):
    print("training model...")
    model.train()
    start = time.time()
    print("opt.epochs",opt.epochs)
    for epoch in range(opt.epochs):
        total_loss = 0
        # print("opt.floyd",opt.floyd)
        # if opt.floyd:
        #     print("   %dm: epoch %d [%s]  %d%%  loss = %s" % ((time.time() - start) // 60, epoch + 1, "".join(' ' * 20), 0, '...'), end='\r')
        #     print("进来了")
        # print(dadas)
        for i, batch in enumerate(opt.train):
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            if opt.device==0:
                src=src.cuda()
                trg_input=trg_input.cuda()

            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()
            total_loss += loss.item()

        print(" epoch %d [%s%s]    loss = %.03f   " %(epoch + 1, "".join('#'*(100//5)), loss))
        if epoch%opt.checkpoint ==0 :
            model_path='weights/model_weights_'+ str(epoch)
            torch.save(model.state_dict(),model_path)
            print("%d models has saved%s" % (epoch, model_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data',default="./data/in.txt")  # 原始数据 初始文件
    parser.add_argument('-trg_data',default="./data/out.txt")  # 原始数据 目标文件
    parser.add_argument('-src_lang',default="en")  # 原始语种
    parser.add_argument('-trg_lang',default="en") #, required=True  # 目标语种
    parser.add_argument('-cuda', default=False, action='store_true')  # 是否使用cuda 添加此将禁用cuda，并运行模型在cpu
    parser.add_argument('-SGDR', action='store_true')  # 增加这将实现随机梯度下降与重启，使用余弦退火
    parser.add_argument('-epochs', type=int, default=1000)  # 要为训练多少个epochs数据(默认值为1000)
    parser.add_argument('-d_model', type=int, default=512)  # 嵌入向量和层的维数(默认为512)
    parser.add_argument('-n_layers', type=int, default=6)  # 在Transformer模型中有多少层(默认=6)
    parser.add_argument('-heads', type=int, default=8)  # 需要分割多少个头部以获得多个头部的注意(默认值为8)
    parser.add_argument('-dropout', type=int, default=0.1)  # 决定多大的dropout将(默认=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)  # 以每次迭代中提供给模型的令牌数(默认值为1500)来度量
    parser.add_argument('-printevery', type=int, default=100)  # 在打印前运行多少次迭代(默认值为100)
    parser.add_argument('-lr', type=int, default=0.0001)  # 学习率(默认值为0.0001)
    parser.add_argument('-premodels',default=False)  # 是否加载原来的权重 和 vecab
    parser.add_argument('-load_weights',default="weights")  # 如果加载预训练的权重，把路径到文件夹，以前的权重和泡菜保存
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=32)  # 判刑与更多的词将不包括在数据集(默认=32)
    parser.add_argument('-floyd', default=True, action='store_true')  # ???????
    parser.add_argument('-checkpoint', type=int, default=100)  # 输入若干分钟。每隔几分钟模型的权重就会被保存到文件夹'weights/'
    #-src_ -trg_data data/english1.txt -src_lang en -trg_lang en -floyd -checkpoint 15 -batchsize 3000 -epochs 10
    opt = parser.parse_args()

    opt.device = 0 if opt.cuda is True else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
    print("opt.device", opt.device)


    read_data(opt)  # 判断数据是否存在 不需要

    SRC, TRG = create_fields(opt)

    opt.train = create_dataset(opt, SRC, TRG)

    print("opt.train",opt.train)

    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
    # print("opt.train_len",opt.train_len)
    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))

    train_model(model, opt)


def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response


def promptNextAction(model, opt, SRC, TRG):
    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0

    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input(
                        "name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res = yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break

            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
                saved_once = 1

            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return


if __name__ == "__main__":
    main()
