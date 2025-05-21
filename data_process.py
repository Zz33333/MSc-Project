import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# 当前文件所在目录
current_dir = os.path.dirname(__file__)
print(current_dir)

def remove_annomaly(file_name):
    df = pd.read_csv(os.path.join(current_dir, f'{file_name}.csv'), delimiter='\t')
    print(df.head(5))

    # 获取Time(s)列，并做异常值处理：'.'符号后应该有三位数，如果小于三位数，需要在数字前面补0
    #Get the Time(s) column and do outlier handling: ‘.’ There should be a three-digit number after the sign,
    # if it is less than three digits, you need to add 0 in front of the number
    time = df['Time(s)']
    time = time.apply(lambda x: x.split('.')[0] + '.' + x.split('.')[1].zfill(3))
    # time中的'.'转为 ':'
    time = time.apply(lambda x: x.replace('.', ':'))
    df['Time(s)'] = time
    print(df.head(30))

    # 将时间表达式转换为秒
    # Converting time expressions to seconds
    def time_to_seconds(time_str):
        h, m, s, ms = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s + ms / 1000
    time = df['Time(s)']
    df['Time(s)'] = df['Time(s)'].apply(time_to_seconds)
    df['Time(s)'] = df['Time(s)'].apply(lambda x: round(x - df.loc[0, 'Time(s)'], 2))
    print(df.head(5))
    
    # time 从0开始重新递增设置，差值为原始数据中的差值
    time_diff = df.loc[1, 'Time(s)'] - df.loc[0, 'Time(s)']
    df['Time(s)'] = np.arange(0, len(df) * time_diff, time_diff)
    df['Time(s)'] = df['Time(s)'].apply(lambda x: round(x, 2))
    print(df.head(5))
    
    # # 如果存在非等差递增的时间戳，修改为等差递增的时间戳：递增量按照第一和第二个时间戳的差值计算
    # time_diff = df['Time(s)'].diff().dropna()
    # time_diff = time_diff[time_diff != time_diff.iloc[0]]
    # if len(time_diff) > 0:
    #     print(f'Non-uniform time interval: {time_diff.iloc[0]}')
    #     df['Time(s)'] = np.arange(0, len(df) * time_diff.iloc[0], time_diff.iloc[0])
    # df['Time(s)'] = time_diff.apply(lambda x: round(x, 2))
    # print(df.head(5))
    # 保存2-8列数据
    df.to_csv(os.path.join(current_dir, f"{file_name}_remove_annomaly.csv"), index=False, columns=df.columns[1:11])

# 以'Time(s)'为横轴绘制除每一列数据
def plot(file_name):
    df = pd.read_csv(os.path.join(current_dir, f'{file_name}.csv'))
    print(df.head(5))

    num_cols = len(df.columns) - 1
    fig, axes = plt.subplots(num_cols, 1, figsize=(30, 5 * num_cols), sharex=True)

    for i, col in enumerate(df.columns):
        if col == 'Time(s)':
            continue
        axes[i-1].plot(df['Time(s)'], df[col], label=col)
        axes[i-1].set_ylabel('Value')
        axes[i-1].legend()
    
    axes[-1].set_xlabel('Time(s)')
    plt.suptitle(f'{file_name} data')
    plt.savefig(os.path.join(current_dir, f'{file_name}_plot.png'))

def plot_df(df, range_start, range_end, title, output_file_name):
    df = df[(df['Time(s)'] >= range_start) & (df['Time(s)'] <= range_end)]
    # print(df.head(5))

    num_cols = len(df.columns) - 1
    fig, axes = plt.subplots(num_cols, 1, figsize=(30, 5 * num_cols), sharex=True)

    if num_cols == 1:
        axes = [axes]
        
    for i, col in enumerate(df.columns):
        if col == 'Time(s)':
            continue
        axes[i-1].plot(df['Time(s)'], df[col], label=col)
        axes[i-1].set_ylabel('Value')
        axes[i-1].legend()
    
    axes[-1].set_xlabel('Time(s)')
    plt.suptitle(title)
    plt.savefig(os.path.join(current_dir, f'{output_file_name}.png'))

# 数据重采样差值填充
# Data resampling difference padding
def resample_fill(df, output_file_name, freq='1S'):
    # 将Time(s)列转换为TimedeltaIndex
    df['Time(s)'] = pd.to_timedelta(df['Time(s)'], unit='s')
    df.set_index('Time(s)', inplace=True)
    # # 删除重复的索引
    # df = df.loc[~df.index.duplicated(keep='first')]
    # 重新采样，每秒的数据，并使用二次样条插值法填充缺失值
    df = df.resample(freq).mean().interpolate()#.interpolate(method='linear')
    print(df.head(5))
    # 重置索引
    df.reset_index(inplace=True)
    print(df.head(5))
    # 转为秒
    df['Time(s)'] = df['Time(s)'].dt.total_seconds()
    # 保存数据
    df.to_csv(os.path.join(current_dir, f"{output_file_name}.csv"), index=False)
    return df

# 定义带通滤波器 Define the bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# 计算并绘制频谱图 Calculate and plot the spectrum
def plot_frequency_spectrum(data, fs, title):
    N = len(data)
    T = 1.0 / fs
    yf = fft(data)
    xf = fftfreq(N, T)[:N//2]

    plt.figure(figsize=(12, 6))
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.savefig(os.path.join(current_dir, f'{title}.png'))

'''IMU数据预处理'''
# 异常数据处理
# remove_annomaly('helmet')
# remove_annomaly('bike')
# 绘制6轴结果
# plot('helmet_remove_annomaly')
# plot('bike_remove_annomaly')
# 上下采样
# upsample_fill('helmet_remove_annomaly', 'helmet_upsample_fill')
# upsample_fill('bike_remove_annomaly', 'bike_upsample_fill')
# plot('helmet_upsample_fill')
# plot('bike_upsample_fill')

GPS_PROCESS = True
IMU_PROCESS = True
OFFSET_PROCESS = True

'''GPS数据处理'''
if GPS_PROCESS:
    # 1.读取GPS数据并获取seconds_elapsed和speed列
    df = pd.read_csv(os.path.join(current_dir, 'location.csv'))
    print(df.head(5))
    df = df[['seconds_elapsed', 'speed']]
    # 修改seconds_elapsed列为'Time(s)'
    df.rename(columns={'seconds_elapsed': 'Time(s)'}, inplace=True)

    # 2.将'Time(s)'为秒，计算平均时间间隔：总时间间隔/总数据量
    sample_time_interval = (df['Time(s)'].iloc[-1] - df['Time(s)'].iloc[0]) / len(df)
    print(f"Sample time interval: {sample_time_interval}")  #1.0290120765754769
    # 从0开始递增设置时间戳，递增量为平均时间间隔
    df['Time(s)'] = df['Time(s)'] - df.loc[0, 'Time(s)']
    print(df.head(10))
    # 可视化原始数据
    plot_df(df, 0, 1000, 'GPS data', 'location_plot')

    # 3.重采样数据为1Hz
    resample_df = resample_fill(df, 'location_resample_fill', freq='100L')
    # 可视化重采样后的数据
    plot_df(resample_df, 0, 8000, 'Resampled GPS data', 'location_resample_fill_plot')
    
    # 4.通过速度计算加速度
    resample_df['ax(g)'] = resample_df['speed'].diff() / sample_time_interval
    # 对初始值进行填充
    resample_df['ax(g)'].iloc[0] = resample_df['ax(g)'].iloc[1]
    # 可视化加速度
    plot_df(resample_df, 0, 8000, 'GPS acceleration', 'location_acceleration')
    # 保存数据
    resample_df.to_csv(os.path.join(current_dir, 'location_resample_fill_with_acceleration.csv'), index=False)

'''IMU数据处理'''
if IMU_PROCESS:
    # 1.IMU预处理
    # 异常数据处理
    remove_annomaly('helmet')
    remove_annomaly('bike')
    # 绘制9轴结果 Plotting 9-axis results
    plot('helmet_remove_annomaly')
    plot('bike_remove_annomaly')
    df_helemet = pd.read_csv(os.path.join(current_dir, 'helmet_remove_annomaly.csv'))
    df_bike = pd.read_csv(os.path.join(current_dir, 'bike_remove_annomaly.csv'))
    
    # # 滤波处理
    # sampling_frequency = 12.5  # 采样频率，单位为Hz
    # # 绘制频谱图
    # plot_frequency_spectrum(df_bike['ax(g)'].values, sampling_frequency, 'Bike x-axis acceleration frequency spectrum')
    # # 设置滤波参数
    # lowcut_frequency = 0.01  # 带通滤波器的低截止频率，单位为Hz
    # highcut_frequency = 6  # 带通滤波器的高截止频率，单位为Hz
    # filter_order = 4  # 滤波器阶数
    # # 对bike x轴加速度进行带通滤波
    # # df_bike['ax(g)'] = bandpass_filter(df_bike['ax(g)'].values, lowcut_frequency, highcut_frequency, sampling_frequency, filter_order)
    # # 绘制滤波后的数据
    # plot_df(df_bike, 0, 7000, 'Bike x-axis acceleration after bandpass filter', 'bike_bandpass_filter')
    
    # 2.重采样数据为10Hz
    resample_df_helmet = resample_fill(df_helemet, 'helmet_resample_fill', freq='100L')
    resample_df_bike = resample_fill(df_bike, 'bike_resample_fill', freq='100L')
    # 可视化重采样后的数据
    plot_df(resample_df_helmet, 0, 7000, 'Resampled helmet data', 'helmet_resample_fill_plot')
    plot_df(resample_df_bike, 0, 7000, 'Resampled bike data', 'bike_resample_fill_plot')

    # 3.根据bike x轴加速度计算速度 Calculate velocity from bike x-axis acceleration
    # 以开头为0的速度为初始速度 Take the velocity that starts with 0 as the initial velocity

    resample_df_bike['speed'] = 0
    sample_time_interval = 0.1

    # 根据x轴加速度找到可能是静止的时间段 Find the time period when it could be stationary based on the x-axis acceleration
    # 速度 = 上一个时刻速度 + 加速度 * 时间间隔
    for i in range(1, len(resample_df_bike)):
        resample_df_bike.loc[i, 'speed'] = resample_df_bike.loc[i-1, 'speed'] + resample_df_bike.loc[i, 'ax(g)'] * sample_time_interval

    # 可视化速度
    plot_df(resample_df_bike, 0, 7000, 'Bike speed', 'bike_speed')
    # 保存数据
    resample_df_bike.to_csv(os.path.join(current_dir, 'bike_resample_fill_with_speed.csv'), index=False)

'''计算GPS和IMU数据的时间差offset'''
if OFFSET_PROCESS:
    # # 读取重采样后的GPS和IMU数据
    # df_bike = pd.read_csv(os.path.join(current_dir, 'bike_resample_fill_with_speed.csv'))
    # df_gps = pd.read_csv(os.path.join(current_dir, 'location_resample_fill.csv'))

    # # 平移IMU数据，使得GPS和IMU数据误差均值(只计算时间重叠部分)最小得到offset
    # offset = 0
    # min_mean_error = float('inf')
    # # 设置offset范围为-100到100s，步长为0.1s
    # for i in range(-1000, 1000):
    #     # 计算误差均值，只计算时间重叠部分
    #     shifted_speed = df_bike['speed'].shift(i)
    #     overlap = df_gps['speed'][shifted_speed.notna()]
    #     error = overlap.values - shifted_speed.dropna().values
    #     mean_error = np.mean(np.abs(error))
    #     if mean_error < min_mean_error:
    #         min_mean_error = mean_error
    #         offset = i
    # print(f"Offset: {offset}, Min mean error: {min_mean_error}")
    
    # 读取重采样后的GPS和IMU数据
    df_bike = pd.read_csv(os.path.join(current_dir, 'bike_resample_fill_with_speed.csv'))
    df_helemet = pd.read_csv(os.path.join(current_dir, 'helmet_resample_fill.csv'))
    df_gps = pd.read_csv(os.path.join(current_dir, 'location_resample_fill_with_acceleration.csv'))

    # 平移IMU数据，使得GPS和IMU数据误差均值(只计算时间重叠部分)最小得到offset
    # Shift the IMU data so that the mean of the GPS and IMU data errors
    # (only the time overlap is calculated) is minimised to get offset
    offset = 0
    min_mean_error = float('inf')
    # 设置offset范围为-100到100s，步长为0.1s Set the offset range from -100 to 100s in 0.1s steps.
    for i in range(-1000, 1000):
        # 计算误差均值，只计算时间重叠部分 Calculate the mean value of the error, only the time overlap is calculated
        shifted_ax = df_bike['ax(g)'].shift(i)
        # 去除bike中为nan的部分 Remove the part of bike that is nan
        shifted_ax = shifted_ax[shifted_ax.notna()]
        # 取bike和gps中最小的index部分进行误差均值计算
        min_index = min(len(shifted_ax), len(df_gps))
        shifted_ax = shifted_ax[:min_index]
        gps_ax = df_gps['ax(g)'][:min_index]
        error = gps_ax.values - shifted_ax.values
        mean_error = np.mean(np.abs(error))
        if mean_error < min_mean_error:
            min_mean_error = mean_error
            offset = i
    print(f"Offset: {offset}, Min mean error: {min_mean_error}")

    # 根据offset对IMU除了第一列以外的数据进行平移，并去除nan值
    for col in df_bike.columns[1:-1]:
        df_bike[col] = df_bike[col].shift(offset)
        df_helemet[col] = df_helemet[col].shift(offset)
    df_bike = df_bike.dropna()
    df_helemet = df_helemet.dropna()
    # 取bike和gps中最小的index部分
    min_index = min(len(df_bike), len(df_gps))
    df_bike = df_bike[:min_index]
    df_helemet = df_helemet[:min_index]
    df_gps = df_gps[:min_index]
    
    # 可视化平移后的bike ax(g)和GPS ax(g)
    plot_df(df_bike, 0, 7000, 'Bike data after offset', 'bike_offset')
    plot_df(df_helemet, 0, 7000, 'Helmet data after offset', 'helmet_offset')
    df_ax_bike_gps = pd.DataFrame({'Time(s)': df_bike['Time(s)'], 'bike_ax(g)': df_bike['ax(g)'], 'gps_ax(g)': df_gps['ax(g)']})
    plot_df(df_ax_bike_gps, 0, 7000, 'Bike and GPS ax(g) after offset', 'bike_gps_offset')
    # 保存数据
    df_bike.to_csv(os.path.join(current_dir, 'bike_offset.csv'), index=False)
    df_helemet.to_csv(os.path.join(current_dir, 'helmet_offset.csv'), index=False)


'''IMU空间旋转对齐'''
# 读取平移后的IMU数据
df_bike = pd.read_csv(os.path.join(current_dir, 'bike_offset.csv'))
df_helemet = pd.read_csv(os.path.join(current_dir, 'helmet_offset.csv'))

# 1.通过三轴角速度找到静止时间点：三轴角速度平方和最小的时间点
df_bike['angular_velocity'] = np.sqrt(df_bike['wx(deg/s)']**2 + df_bike['wy(deg/s)']**2 + df_bike['wz(deg/s)']**2)
df_helemet['angular_velocity'] = np.sqrt(df_helemet['wx(deg/s)']**2 + df_helemet['wy(deg/s)']**2 + df_helemet['wz(deg/s)']**2)
# 找到静止时间点
static_index_bike = df_bike['angular_velocity'].idxmin()
static_index_helemet = df_helemet['angular_velocity'].idxmin()
print(f"Static index bike: {static_index_bike}, Static index helmet: {static_index_helemet}")

# 2.计算初始欧拉角
# 以静止时间点为初始欧拉角，计算俯仰角(pitch)、横滚角(roll)和偏航角(yaw) 
# AngleX为俯仰角，AngleY为横滚角，AngleZ为偏航角


# 3.根据初始欧拉角对IMU进行姿态解算（IMU相对于导航坐标系的三轴角度）

# 4.根据IMU相对于导航坐标系的三轴角度 对IMU数据进行旋转对齐
# 计算每个时间点载体坐标系到导航坐标系的姿态旋转矩阵
# 旋转矩阵公式：R = Rz(yaw) * Ry(pitch) * Rx(roll) = 
# [cos(yaw)cos(pitch), cos(yaw)sin(pitch)sin(roll)-sin(yaw)cos(roll), cos(yaw)sin(pitch)cos(roll)+sin(yaw)sin(roll); 
#  sin(yaw)cos(pitch), sin(yaw)sin(pitch)sin(roll)+cos(yaw)cos(roll), sin(yaw)sin(pitch)cos(roll)-cos(yaw)sin(roll); 
#  -sin(pitch),        cos(pitch)sin(roll),                           cos(pitch)cos(roll)]

# 5.根据旋转矩阵对IMU数据进行旋转对齐
# 旋转矩阵对向量（单个时间点）进行旋转：v' = R * v ;  v = (ax, ay, az)^T或(wx, wy, wz)^T
# 循环遍历每个时间点，对IMU数据的加速度和角速度进行旋转对齐
for i in range(len(df_bike)):
    # bike
    # 计算旋转矩阵
    yaw, pitch, roll = np.deg2rad([df_bike.loc[i, 'AngleZ(deg)'], df_bike.loc[i, 'AngleX(deg)'], df_bike.loc[i, 'AngleY(deg)']])
    R = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                  [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                  [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
    # 对加速度和角速度进行旋转对齐
    acc = np.dot(R, df_bike.loc[i, ['ax(g)', 'ay(g)', 'az(g)']])
    df_bike.loc[i, ['ax(g)', 'ay(g)', 'az(g)']] = acc

    gyro = np.dot(R, df_bike.loc[i, ['wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']])
    df_bike.loc[i, ['wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']] = gyro
    
    # helmet
    # 计算旋转矩阵
    yaw, pitch, roll = np.deg2rad([df_helemet.loc[i, 'AngleZ(deg)'], df_helemet.loc[i, 'AngleX(deg)'], df_helemet.loc[i, 'AngleY(deg)']])
    R = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                  [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                  [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
    # 对加速度和角速度进行旋转对齐
    acc = np.dot(R, df_helemet.loc[i, ['ax(g)', 'ay(g)', 'az(g)']])
    df_helemet.loc[i, ['ax(g)', 'ay(g)', 'az(g)']] = acc
    
    gyro = np.dot(R, df_helemet.loc[i, ['wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']])
    df_helemet.loc[i, ['wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']] = gyro
    
    
# 可视化旋转对齐后的数据
plot_df(df_bike, 0, 7000, 'Bike data after rotation alignment', 'bike_rotation_alignment')
plot_df(df_helemet, 0, 7000, 'Helmet data after rotation alignment', 'helmet_rotation_alignment')
# 保存数据
df_bike.to_csv(os.path.join(current_dir, 'bike_rotation_alignment.csv'), index=False)
df_helemet.to_csv(os.path.join(current_dir, 'helmet_rotation_alignment.csv'), index=False)








    

