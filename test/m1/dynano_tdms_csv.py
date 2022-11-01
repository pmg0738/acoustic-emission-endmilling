#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import codecs
import array
import sys
import pandas as pd


# In[2]:


def load_tdms(path, ch_num):
    ch = [[] for _ in range(ch_num)]
    with codecs.open(path, 'rb') as f:
        while True:
            # リードインと呼ばれる部分の読み込み
            tdms = f.read(28) 
            # print(f.tell())
            # print(tdms)
            #if b'TDSm' != tdms[:4]:
            #    continue

            # ファイルを最後まで読み込んだら終了
            if tdms==b'': 
                break

            # データとその区切りを取得
            # 次のセグメントの位置を抽出
            seg_ofs = tdms[12:20]
            by1 = array.array('l')
            by1.frombytes( seg_ofs )
            seg_ofs = np.asarray(by1)[0]
            # データの位置を抽出
            data_ofs= tdms[20:28] 
            by2 = array.array('l')
            by2.frombytes( data_ofs )
            data_ofs = np.asarray(by2)[0]

            # データのある部分まで読み飛ばす
            tdms = f.read(data_ofs)
            #print(tdms[:200], len(tdms))
            #if len(ch[0])==2:
            #    exit()
            #if len(ch[0])==0:
            #    head=tdms
            # データ部分の読み込み
            tdms = f.read(seg_ofs-data_ofs)
            by = array.array('f')
            by.frombytes( tdms )
            data = np.asarray( by )
            # print(data.shape)

            # 各チャンネルを取得
            for i in range(ch_num):
                ch[i].append( data[i::ch_num].reshape(-1,1) )

    for i in range(ch_num):
        ch[i] = np.vstack(ch[i])[:,0]

    return ch


# In[3]:


if __name__ == '__main__':
    path = r'C:\Users\pmg07\M1\pr1\Dynamometer_127.tdms'
    # ch_numにはチャンネル数を渡す. AE_Signal, AE_Noise, AE_AF で3チャンネル.
    # data[0]にAE_Signal, data[1]にAE_Noise, data[2]にAE_AFのデータが入る.
    data = load_tdms(path, ch_num=3)
    # 例：各データの前方・後方10データずつ表示
    print( data[0][:10], data[0][-10:] )
    print( data[1][:10], data[1][-10:] )
    print( data[2][:10], data[2][-10:] )


# In[4]:


len(data[0])


# In[44]:


time_length = 16.668

# 時間を表す配列を作る
time_d = []
# 1サンプリングするのにかかる時間
sample_rate = 1/(10**6)*3

# 時間を表す配列を作るために、16,548,822(flatten_all_dataの長さ)分だけ
# sample_rate × i をする。ただし、16.668(time_length)を超えない時まで
for i in range(1, len(data[0])+1):
    if(time_length < sample_rate*(i-1)):
        break
    time_d.append(sample_rate*(i-1))


# In[51]:


dict1 = dict(time=time_d[:3000000], x=data[0][:3000000], y=data[1][:3000000], z=data[2][:3000000])
dict2 = dict(time=time_d[3000000:], x=data[0][3000000:], y=data[1][3000000:], z=data[2][3000000:])


# In[54]:


print(len(dict2['time']))
print(len(dict2['z']))


# In[55]:


df1 = pd.DataFrame(data=dict1)
df2 = pd.DataFrame(data=dict2)


# In[56]:


df1.to_csv('Dynano_tdms_to_csv_with_ishidaSan.csv')
df2.to_csv('Dynano_tdms_to_csv_with_ishidaSan.csv', mode='a', header=False)


# In[ ]:




