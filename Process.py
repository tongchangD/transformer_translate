import pandas as pd
import torchtext
from torchtext import data
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle
import spacy
import re


class tokenize(object):

    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        print("sentence0",sentence)
        sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        print("sentence1",sentence)
        print("tokenize_return",[tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "])
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


def read_data(opt):
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data).read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()

    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data).read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()


def create_fields(opt):
    spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl']
    if opt.src_lang not in spacy_langs:
        print('invalid src language: ' + opt.src_lang + 'supported languages : ' + spacy_langs)
    if opt.trg_lang not in spacy_langs:
        print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + spacy_langs)

    print("loading spacy tokenizers...")

    t_src = tokenize(opt.src_lang)
    t_trg = tokenize(opt.trg_lang)

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer,init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)
    print("TRG",TRG)
    print("SRC",SRC)
    print("opt.premodels",opt.premodels)
    if opt.premodels:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("11error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
    return (SRC, TRG)


def create_dataset(opt, SRC, TRG):
    print("creating dataset and iterator... ")

    raw_data = {'src': [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    print("raw_data",raw_data)
    # 此处开始制作 一个 csv 文件
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    # print("opt.max_strlen",opt.max_strlen)

    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]
    df.to_csv("translate_transformer_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)
    print("train_iter",train_iter)
    os.remove('translate_transformer_temp.csv')
    # 此处 删除 制作的 csv 文件

    if not opt.premodels :  # 加载权重
        # print("SRC.vocab.opt.premodels",SRC.vocab)
        # print("TRG.vocab.opt.premodels",TRG.vocab)
        SRC.build_vocab(train)
        TRG.build_vocab(train)
        if opt.checkpoint > 0:
            if not os.path.exists(opt.load_weights):
                os.mkdir(opt.load_weights)
                print("weights folder already exists, run program with -load_weights weights to load them")
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    print("SRC.vocab",len(SRC.vocab),SRC.vocab.freqs)
    print("TRG.vocab",len(TRG.vocab),TRG.vocab.freqs)

    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    # opt.train_len = get_len(train_iter)
    # print("train_len",opt.train_len)
    print("train_iter",train_iter)
    return train_iter


# def get_len(train):
#     for i, b in enumerate(train):
#         pass
#     return i
