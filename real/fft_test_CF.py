# from loadTDMS import *
import numpy as np
import matplotlib.pyplot as plt


# 関数の集まり
def FFT_main(t, x, dt, split_t_r, overlap, window_F):

    # データをオーバーラップして分割する。
    split_data = data_split(t, x, split_t_r, overlap)

    # FFTを行う。
    FFT_result_list = []
    for split_data_cont in split_data:
        FFT_result_cont = FFT(split_data_cont, dt, window_F)
        FFT_result_list.append(FFT_result_cont)

    # 平均化
    fq_ave = FFT_result_list[0][0]
    F_abs_amp_ave = np.zeros(len(fq_ave))
    for i in range(len(FFT_result_list)):
        F_abs_amp_ave = F_abs_amp_ave + FFT_result_list[i][1]
    F_abs_amp_ave = F_abs_amp_ave/(i+1)

#     plot_FFT(t, x, fq_ave, F_abs_amp_ave, output_FN, "ave", 1, y_label, y_unit)

    return fq_ave, F_abs_amp_ave


def FFT(data_input, dt, window_F):

    N = len(data_input[0])
    
    # 窓の用意
    if window_F == "hanning":
        window = np.hanning(N)          # ハニング窓
    elif window_F == "hamming":
        window = np.hamming(N)          # ハミング窓
    elif window_F == "blackman":
        window = np.blackman(N)         # ブラックマン窓
    else:
        print("Error: input window function name is not sapported. Your input: ", window_F)
        print("Hanning window function is used.")
        hanning = np.hanning(N)          # ハニング窓

        
#     print(data_input[0].shape)
#     print(data_input[1].shape)
#     print(window.shape)

    # 窓関数後の信号
    x_windowed = data_input[1]*window

    # FFT計算
    F = np.fft.fft(x_windowed)
    F_abs = np.abs(F)
    F_abs_amp = F_abs / N * 2
    fq = np.linspace(0, 1.0/dt, N)
#     fq = np.linspace(0, N*dt, N)

    # 窓補正
    acf = 1/(sum(window)/N)
    F_abs_amp = acf*F_abs_amp

    # ナイキスト定数まで抽出
    fq_out = fq[:int(N/2)+1]
    F_abs_amp_out = F_abs_amp[:int(N/2)+1]

    return [fq_out, F_abs_amp_out]


def data_split(t, x, split_t_r, overlap):

    split_data = []
    one_frame_N = int(len(t)*split_t_r)  # 1フレームのサンプル数
    overlap_N = int(one_frame_N*overlap)  # オーバーラップするサンプル数
    start_S = 0
    end_S = start_S + one_frame_N

    while True:
        t_cont = t[start_S:end_S]
        x_cont = x[start_S:end_S]
        split_data.append([t_cont, x_cont])

        start_S = start_S + (one_frame_N - overlap_N)
        end_S = start_S + one_frame_N

        if end_S > len(t):
            break

    return np.array(split_data)


def plot_FFT(fq, F_abs_amp, output_FN, y_label, y_unit, num):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.xlabel('freqency[kHz]', fontsize=16)

    plt.ylabel(y_label+"[dB]", fontsize=16)

#     ax.set_xlim(0, 500000)
#     ax.set_xticklabels([0, 100, 200, 300, 400, 500])
#     ax.set_ylim(-40, 50)
#     ax.set_yticklabels([-40,-20, 0, 20, 40])

    plt.title(rf'{num}path')
    
    plt.plot(fq, 20 * np.log10(F_abs_amp))
    
    ax.grid()
    p = plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(output_FN, dpi=300)
    plt.close()

    return 0

def plot_FFT_minmax(fq, F_abs_amp, output_FN, y_label, y_unit, num):

    fig = plt.figure()
    ax = fig.add_subplot(111)

#     plt.xlabel('freqency[kHz]', fontsize=16)
#     plt.ylabel(y_label+"[dB]", fontsize=16)
    plt.title(rf'{num}path')
    plt.plot(min_max_p(fq), min_max_p(20 * np.log10(F_abs_amp)))
#     plt.plot(min_max_p(20 * np.log10(F_abs_amp)))

    
    ax.grid()
    p = plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(output_FN, dpi=300)
    plt.close()

    return 0




# 正規化を行う関数の定義
def min_max_p(p):
  #最小値の計算
  min_p = p.min()
  #最大値の計算
  max_p = p.max()
  #正規化の計算
  min_max_p = (p - min_p) / (max_p - min_p)
  return min_max_p












# # from loadTDMS import *
# import numpy as np
# import matplotlib.pyplot as plt


# def FFT_main(t, x, dt, split_t_r, overlap, window_F):

#     # データをオーバーラップして分割する。
#     split_data = data_split(t, x, split_t_r, overlap)

#     # FFTを行う。
#     FFT_result_list = []
#     for split_data_cont in split_data:
#         FFT_result_cont = FFT(split_data_cont, dt, window_F)
#         FFT_result_list.append(FFT_result_cont)

#     # 平均化
#     fq_ave = FFT_result_list[0][0]
#     F_abs_amp_ave = np.zeros(len(fq_ave))
#     for i in range(len(FFT_result_list)):
#         F_abs_amp_ave = F_abs_amp_ave + FFT_result_list[i][1]
#         F_abs_amp_ave = F_abs_amp_ave/(i+1)

# #     plot_FFT(t, x, fq_ave, F_abs_amp_ave, output_FN, "ave", 1, y_label, y_unit)

#     return fq_ave, F_abs_amp_ave




# def FFT(data_input, dt, window_F):

#     N = len(data_input[0])

#     # 窓の用意
#     if window_F == "hanning":
#         window = np.hanning(N)          # ハニング窓
#     elif window_F == "hamming":
#         window = np.hamming(N)          # ハミング窓
#     elif window_F == "blackman":
#         window = np.blackman(N)         # ブラックマン窓
#     else:
#         print("Error: input window function name is not sapported. Your input: ", window_F)
#         print("Hanning window function is used.")
#         hanning = np.hanning(N)          # ハニング窓

#     # 窓関数後の信号
#     x_windowed = data_input[1]*window

#     # FFT計算
#     F = np.fft.fft(x_windowed)
#     F_abs = np.abs(F)
#     F_abs_amp = F_abs / N * 2
#     fq = np.linspace(0, 1.0/dt, N)
# #     fq = np.linspace(0, N*dt, N)

#     # 窓補正
#     acf = 1/(sum(window)/N)
#     F_abs_amp = acf*F_abs_amp

#     # ナイキスト定数まで抽出
#     fq_out = fq[:int(N/2)+1]
#     F_abs_amp_out = F_abs_amp[:int(N/2)+1]

#     return [fq_out, F_abs_amp_out]


# def data_split(t, x, split_t_r, overlap):

#     split_data = []
#     one_frame_N = int(len(t)*split_t_r)  # 1フレームのサンプル数
#     overlap_N = int(one_frame_N*overlap)  # オーバーラップするサンプル数
#     start_S = 0
#     end_S = start_S + one_frame_N

#     while True:
#         t_cont = t[start_S:end_S]
#         x_cont = x[start_S:end_S]
#         split_data.append([t_cont, x_cont])

#         start_S = start_S + (one_frame_N - overlap_N)
#         end_S = start_S + one_frame_N

#         if end_S > len(t):
#             break

#     return np.array(split_data)


# def plot_FFT(fq, F_abs_amp, output_FN, y_label, y_unit):

#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     plt.xlabel('freqency[Hz]', fontsize=16)

#     plt.ylabel(y_label+"[dB]", fontsize=16)

#     ax.set_xlim(0, 500000)
#     ax.set_xticklabels([0, 100, 200, 300, 400, 500])
#     ax.set_ylim(-160, -40)
#     ax.set_yticklabels([-160,-140,-120,-100, -80, -60, -40])

#     plt.plot(fq, 20 * np.log10(F_abs_amp))
#     ax.grid()
#     p = plt.tick_params(labelsize=16)
#     plt.tight_layout()
#     plt.savefig(output_FN, dpi=300)

#     return 0

# def plot_sub_FFT(fq, F_abs_amp, F_abs_amp2, output_FN, y_label, y_unit):

#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     plt.xlabel('freqency[kHz]', fontsize=16)

#     plt.ylabel(y_label+"[dB]", fontsize=16)

#     ax.set_xlim(0, 500000)
#     ax.set_xticklabels([0, 100, 200, 300, 400, 500])
#     ax.set_ylim(-100, -20)
#     ax.set_yticklabels([-100, -80, -60, -40, -20])

#     plt.plot(fq, 20 * np.log10(F_abs_amp) - 20 * np.log10(F_abs_amp2))

#     p = plt.tick_params(labelsize=16)
#     plt.tight_layout()
#     plt.savefig(output_FN, dpi=300)

#     return 0

# def FFT_substract(amp1, amp2):
#     # 基準がamp2
#     return amp1 - amp2
