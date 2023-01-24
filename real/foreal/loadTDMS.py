import codecs
import numpy as np
import array
# import sys
# import os


def load_tdms(path, ch_num):
    ch = [[] for _ in range(ch_num)]
    with codecs.open(path, 'rb') as f:
        while True:
            # リードインと呼ばれる部分の読み込み
            tdms = f.read(28)
            # print(f.tell())
            # print(tdms)
            # if b'TDSm' != tdms[:4]:
            #    continue

            # ファイルを最後まで読み込んだら終了
            if tdms == b'':
                break

            # データとその区切りを取得
            # 次のセグメントの位置を抽出
            seg_ofs = tdms[12:20]
            by1 = array.array('l')
            by1.frombytes(seg_ofs)
            seg_ofs = np.asarray(by1)[0]
            # データの位置を抽出
            data_ofs = tdms[20:28]
            by2 = array.array('l')
            by2.frombytes(data_ofs)
            data_ofs = np.asarray(by2)[0]

            # データのある部分まで読み飛ばす
            tdms = f.read(data_ofs)
            #print(tdms[:200], len(tdms))
            # if len(ch[0])==2:
            #    exit()
            # if len(ch[0])==0:
            #    head=tdms
            # データ部分の読み込み
            tdms = f.read(seg_ofs-data_ofs)
            by = array.array('f')
            by.frombytes(tdms)
            data = np.asarray(by)
            # print(data.shape)

            # 各チャンネルを取得
            for i in range(ch_num):
                ch[i].append(data[i::ch_num].reshape(-1, 1))

    for i in range(ch_num):
        ch[i] = np.vstack(ch[i])[:, 0]

    return ch
