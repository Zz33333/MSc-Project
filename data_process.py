import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# The directory where the current file is located
current_dir = os.path.dirname(__file__)
print(current_dir)

def remove_annomaly(file_name):
    df = pd.read_csv(os.path.join(current_dir, f'{file_name}.csv'), delimiter='\t')
    print(df.head(5))

    # Get the Time(s) column and handle outliers: There should be three digits after the '.' symbol.
    # If it is less than three digits, add 0 in front of the number.
    time = df['Time(s)']
    time = time.apply(lambda x: x.split('.')[0] + '.' + x.split('.')[1].zfill(3))
    # Convert '.' in time to ':'
    time = time.apply(lambda x: x.replace('.', ':'))
    df['Time(s)'] = time
    print(df.head(30))

    # Converts a time expression to seconds
    def time_to_seconds(time_str):
        h, m, s, ms = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s + ms / 1000
    time = df['Time(s)']
    df['Time(s)'] = df['Time(s)'].apply(time_to_seconds)
    df['Time(s)'] = df['Time(s)'].apply(lambda x: round(x - df.loc[0, 'Time(s)'], 2))
    print(df.head(5))
    
    # time is reset from 0, and the difference is the difference in the original data
    time_diff = df.loc[1, 'Time(s)'] - df.loc[0, 'Time(s)']
    df['Time(s)'] = np.arange(0, len(df) * time_diff, time_diff)
    df['Time(s)'] = df['Time(s)'].apply(lambda x: round(x, 2))
    print(df.head(5))
    
    # # If there are non-uniformly increasing timestamps, change them to uniformly increasing timestamps:
    # the increment is calculated based on the difference between the first and second timestamps.
    # time_diff = df['Time(s)'].diff().dropna()
    # time_diff = time_diff[time_diff != time_diff.iloc[0]]
    # if len(time_diff) > 0:
    #     print(f'Non-uniform time interval: {time_diff.iloc[0]}')
    #     df['Time(s)'] = np.arange(0, len(df) * time_diff.iloc[0], time_diff.iloc[0])
    # df['Time(s)'] = time_diff.apply(lambda x: round(x, 2))
    # print(df.head(5))
    # Save 2-8 columns of data
    df.to_csv(os.path.join(current_dir, f"{file_name}_remove_annomaly.csv"), index=False, columns=df.columns[1:11])

# Plot each column of data using 'Time(s)' as the horizontal axis
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

# Data resampling difference filling
def resample_fill(df, output_file_name, freq='1S'):
    # Convert Time(s) column to TimedeltaIndex
    df['Time(s)'] = pd.to_timedelta(df['Time(s)'], unit='s')
    df.set_index('Time(s)', inplace=True)
    # # Remove duplicate indexes
    # df = df.loc[~df.index.duplicated(keep='first')]

    df = df.resample(freq).mean().interpolate()#.interpolate(method='linear')
    print(df.head(5))
    # Reset Index
    df.reset_index(inplace=True)
    print(df.head(5))
    # Convert to seconds
    df['Time(s)'] = df['Time(s)'].dt.total_seconds()
    # save the data
    df.to_csv(os.path.join(current_dir, f"{output_file_name}.csv"), index=False)
    return df

# Defining a Bandpass Filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Calculate and plot the spectrum
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

'''IMU data preprocessing'''
# Abnormal data processing
# remove_annomaly('helmet')
# remove_annomaly('bike')
# Plot 6-axis results
# plot('helmet_remove_annomaly')
# plot('bike_remove_annomaly')
# Up and down sampling
# upsample_fill('helmet_remove_annomaly', 'helmet_upsample_fill')
# upsample_fill('bike_remove_annomaly', 'bike_upsample_fill')
# plot('helmet_upsample_fill')
# plot('bike_upsample_fill')

GPS_PROCESS = False
IMU_PROCESS = False
OFFSET_PROCESS = True
IMU_ROTATION_ALIGNMENT = False





'''GPS Data Processing'''
if GPS_PROCESS:
    # 1.Read GPS data and get the seconds_elapsed and speed columns
    df = pd.read_csv(os.path.join(current_dir, 'location.csv'))
    print(df.head(5))
    df = df[['seconds_elapsed', 'speed', 'latitude', 'longitude','bearing']]
    # 修改seconds_elapsed列为'Time(s)'
    df.rename(columns={'seconds_elapsed': 'Time(s)'}, inplace=True)

    # 2.Set 'Time(s)' to seconds and calculate the average time interval: total time interval/total data volume
    sample_time_interval = (df['Time(s)'].iloc[-1] - df['Time(s)'].iloc[0]) / len(df)
    print(f"Sample time interval: {sample_time_interval}")  #1.0290120765754769
    # Set the timestamp starting from 0 and incrementing by the average time interval
    df['Time(s)'] = df['Time(s)'] - df.loc[0, 'Time(s)']
    print(df.head(10))
    # Visualizing raw data
    plot_df(df, 0, 1000, 'GPS data', 'location_plot')

    # 3.Resample data to 1Hz
    resample_df = resample_fill(df, 'location_resample_fill', freq='100L')
    # Visualizing the resampled data
    plot_df(resample_df, 0, 8000, 'Resampled GPS data', 'location_resample_fill_plot')
    
    # 4.Calculate acceleration from velocity
    resample_df['ax(g)'] = resample_df['speed'].diff() / sample_time_interval
    # Fill in the initial value
    resample_df['ax(g)'].iloc[0] = resample_df['ax(g)'].iloc[1]
    # Visualizing Acceleration
    plot_df(resample_df, 0, 8000, 'GPS acceleration', 'location_acceleration')
    # save the data
    resample_df.to_csv(os.path.join(current_dir, 'location_resample_fill_with_acceleration.csv'), index=False)

'''IMU Data Processing'''
if IMU_PROCESS:
    # 1.IMU preprocessing
    # # Abnormal data processing
    remove_annomaly('helmet')
    remove_annomaly('bike')
    # Plotting 9-axis results
    plot('helmet_remove_annomaly')
    plot('bike_remove_annomaly')
    df_helemet = pd.read_csv(os.path.join(current_dir, 'helmet_remove_annomaly.csv'))
    df_bike = pd.read_csv(os.path.join(current_dir, 'bike_remove_annomaly.csv'))
    
    # # Filtering
    # sampling_frequency = 12.5  # Sampling frequency, in Hz
    # # Plotting the Spectrum
    # plot_frequency_spectrum(df_bike['ax(g)'].values, sampling_frequency, 'Bike x-axis acceleration frequency spectrum')
    # # Set filter parameters
    # lowcut_frequency = 0.01  # Low cutoff frequency of the bandpass filter, in Hz
    # # highcut_frequency = 6 # High cutoff frequency of the bandpass filter, in Hz
    # filter_order = 4  #
    # # Bandpass filter the bike x-axis acceleration
    # # df_bike['ax(g)'] = bandpass_filter(df_bike['ax(g)'].values, lowcut_frequency, highcut_frequency, sampling_frequency, filter_order)
    # # Plotting the filtered data
    # plot_df(df_bike, 0, 7000, 'Bike x-axis acceleration after bandpass filter', 'bike_bandpass_filter')
    
    # 2.Resample data to 1Hz
    resample_df_helmet = resample_fill(df_helemet, 'helmet_resample_fill', freq='100L')
    resample_df_bike = resample_fill(df_bike, 'bike_resample_fill', freq='100L')
    # Visualizing the resampled data
    plot_df(resample_df_helmet, 0, 7000, 'Resampled helmet data', 'helmet_resample_fill_plot')
    plot_df(resample_df_bike, 0, 7000, 'Resampled bike data', 'bike_resample_fill_plot')

    # 3.Calculate the speed based on the bike's x-axis acceleration
    # # The speed starting with 0 is the initial speed
    resample_df_bike['speed'] = 0
    sample_time_interval = 0.1

    # Find the possible static time period based on the x-axis acceleration
    # # Speed = previous moment speed + acceleration * time interval v = v0+ at
    for i in range(1, len(resample_df_bike)):
        resample_df_bike.loc[i, 'speed'] = resample_df_bike.loc[i-1, 'speed'] + resample_df_bike.loc[i, 'ax(g)'] * sample_time_interval

    # Visualize speed
    plot_df(resample_df_bike, 0, 7000, 'Bike speed', 'bike_speed')
    # save the data
    resample_df_bike.to_csv(os.path.join(current_dir, 'bike_resample_fill_with_speed.csv'), index=False)

'''Calculate the time difference offset between GPS and IMU data'''
if OFFSET_PROCESS:
    # Read resampled GPS and IMU data
    df_bike = pd.read_csv(os.path.join(current_dir, 'bike_resample_fill_with_speed.csv'))
    df_helemet = pd.read_csv(os.path.join(current_dir, 'helmet_resample_fill.csv'))
    df_gps = pd.read_csv(os.path.join(current_dir, 'location_resample_fill_with_acceleration.csv'))

    # Translate the IMU data so that the mean error between GPS and IMU data (only the time overlap is calculated) is minimized to get the offset
    offset = 0
    min_mean_error = float('inf')
    # Set the offset range to -100 to 100s, with a step size of 0.1s
    for i in range(-1000, 1000):
        # Calculate the error mean, only the time overlap part
        shifted_ax = df_bike['ax(g)'].shift(i)
        # Remove the nan part of bike
        shifted_ax = shifted_ax[shifted_ax.notna()]
        # Take the smallest index part of bike and GPS to calculate the error mean
        min_index = min(len(shifted_ax), len(df_gps))
        shifted_ax = shifted_ax[:min_index]
        gps_ax = df_gps['ax(g)'][:min_index]
        error = gps_ax.values - shifted_ax.values
        mean_error = np.mean(np.abs(error))
        if mean_error < min_mean_error:
            min_mean_error = mean_error
            offset = i
    print(f"Offset: {offset}, Min mean error: {min_mean_error}")
    
    # According to the offset, translate the IMU data except the first column and remove the nan value
    for col in df_bike.columns[1:-1]:
        df_bike[col] = df_bike[col].shift(offset)
        df_helemet[col] = df_helemet[col].shift(offset)
    df_bike = df_bike.dropna()
    df_helemet = df_helemet.dropna()
    # Take the smallest index part of bike and gps
    min_index = min(len(df_bike), len(df_gps))
    df_bike = df_bike[:min_index]
    df_helemet = df_helemet[:min_index]
    df_gps = df_gps[:min_index]
    
    # Visualize the translated bike ax(g) and GPS ax(g)
    plot_df(df_bike, 0, 7000, 'Bike data after offset', 'bike_offset')
    plot_df(df_helemet, 0, 7000, 'Helmet data after offset', 'helmet_offset')
    df_ax_bike_gps = pd.DataFrame({'Time(s)': df_bike['Time(s)'], 'bike_ax(g)': df_bike['ax(g)'], 'gps_ax(g)': df_gps['ax(g)']})
    plot_df(df_ax_bike_gps, 0, 7000, 'Bike and GPS ax(g) after offset', 'bike_gps_offset')
    # save the data
    df_bike.to_csv(os.path.join(current_dir, 'bike_offset.csv'), index=False)
    df_helemet.to_csv(os.path.join(current_dir, 'helmet_offset.csv'), index=False)
    df_gps.to_csv(os.path.join(current_dir, 'location_offset.csv'), index=False)

'''IMU space rotation alignment'''
if IMU_ROTATION_ALIGNMENT:
    # Read the IMU data after translation
    df_bike = pd.read_csv(os.path.join(current_dir, 'bike_offset.csv'))
    df_helemet = pd.read_csv(os.path.join(current_dir, 'helmet_offset.csv'))

    # 1.Find the stationary time point through the three-axis angular velocity:
    # the time point when the sum of the squares of the three-axis angular velocity is the smallest
    df_bike['angular_velocity'] = np.sqrt(df_bike['wx(deg/s)']**2 + df_bike['wy(deg/s)']**2 + df_bike['wz(deg/s)']**2)
    df_helemet['angular_velocity'] = np.sqrt(df_helemet['wx(deg/s)']**2 + df_helemet['wy(deg/s)']**2 + df_helemet['wz(deg/s)']**2)
    # Finding a stationary point in time
    static_index_bike = df_bike['angular_velocity'].idxmin()
    static_index_helemet = df_helemet['angular_velocity'].idxmin()
    print(f"Static index bike: {static_index_bike}, Static index helmet: {static_index_helemet}")

    # 2.Calculate initial Euler angles
    # # Take the static time point as the initial Euler angle, calculate the pitch angle (pitch), roll angle (roll) and yaw angle (yaw)
    # # AngleX is the pitch angle, AngleY is the roll angle, AngleZ is the yaw angle

    # 3.Solve the attitude of IMU according to the initial Euler angle (the three-axis angle of IMU relative to the navigation coordinate system)
    #
    # # 4. Rotate and align the IMU data according to the three-axis angle of IMU relative to the navigation coordinate system
    # # Calculate the attitude rotation matrix from the carrier coordinate system to the navigation coordinate system at each time point
    # # Rotation matrix formula: R = Rz(yaw) * Ry(pitch) * Rx(roll) =
    # [cos(yaw)cos(pitch), cos(yaw)sin(pitch)sin(roll)-sin(yaw)cos(roll), cos(yaw)sin(pitch)cos(roll)+sin(yaw)sin(roll); 
    #  sin(yaw)cos(pitch), sin(yaw)sin(pitch)sin(roll)+cos(yaw)cos(roll), sin(yaw)sin(pitch)cos(roll)-cos(yaw)sin(roll); 
    #  -sin(pitch),        cos(pitch)sin(roll),                           cos(pitch)cos(roll)]

    # 5.Rotate and align IMU data according to the rotation matrix
    # # The rotation matrix rotates the vector (single time point)：v' = R * v ;  v = (ax, ay, az)^T or (wx, wy, wz)^T
    # Loop through each time point and rotate and align the acceleration and angular velocity of the IMU data
    for i in range(len(df_bike)):
        # bike
        # Calculate the rotation matrix
        yaw, pitch, roll = np.deg2rad([df_bike.loc[i, 'AngleZ(deg)'], df_bike.loc[i, 'AngleX(deg)'], df_bike.loc[i, 'AngleY(deg)']])
        R = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                    [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                    [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
        # Rotationally align acceleration and angular velocity
        acc = np.dot(R, df_bike.loc[i, ['ax(g)', 'ay(g)', 'az(g)']])
        df_bike.loc[i, ['ax(g)', 'ay(g)', 'az(g)']] = acc

        gyro = np.dot(R, df_bike.loc[i, ['wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']])
        df_bike.loc[i, ['wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']] = gyro
        
        # helmet
        # Calculate the rotation matrix
        yaw, pitch, roll = np.deg2rad([df_helemet.loc[i, 'AngleZ(deg)'], df_helemet.loc[i, 'AngleX(deg)'], df_helemet.loc[i, 'AngleY(deg)']])
        R = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                    [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                    [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
        # Rotationally align acceleration and angular velocity
        acc = np.dot(R, df_helemet.loc[i, ['ax(g)', 'ay(g)', 'az(g)']])
        df_helemet.loc[i, ['ax(g)', 'ay(g)', 'az(g)']] = acc
        
        gyro = np.dot(R, df_helemet.loc[i, ['wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']])
        df_helemet.loc[i, ['wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']] = gyro
        
    # Visualizing the Rotated Aligned Data
    plot_df(df_bike, 0, 7000, 'Bike data after rotation alignment', 'bike_rotation_alignment')
    plot_df(df_helemet, 0, 7000, 'Helmet data after rotation alignment', 'helmet_rotation_alignment')
    # save the data
    df_bike.to_csv(os.path.join(current_dir, 'bike_rotation_alignment.csv'), index=False)
    df_helemet.to_csv(os.path.join(current_dir, 'helmet_rotation_alignment.csv'), index=False)



'''Calculate the acceleration in the x-axis and y-axis directions using the longitude and latitude data from GPS'''
# Read GPS data
df_gps = pd.read_csv(os.path.join(current_dir, 'location.csv'))
# Calculate the acceleration in the x-axis (east) and y-axis (north) directions using the GPS longitude and latitude data
# 1. Calculate the velocity components in the x- and y-directions based on the longitude and latitude of adjacent time points
# Calculate the distance in the x- and y-directions between two time points: distance in the x-direction = east longitude difference * 111.3195 * cos(latitude); distance in the y-direction = latitude difference * 111.3195;
# 111.3195km is the distance corresponding to 1 degree of latitude
x_distance = (df_gps['longitude'].diff() * 111319.5 * np.cos(np.deg2rad(df_gps['latitude']))).fillna(0)
y_distance = (df_gps['latitude'].diff() * 111319.5).fillna(0)
theta = np.arctan2(y_distance, x_distance)
# 2.Calculate the speed in the x-axis and y-axis directions between two time points
x_speed = df_gps['speed'] * np.cos(theta)
y_speed = df_gps['speed'] * np.sin(theta)
# 3.Calculate the acceleration in the x-axis and y-axis directions between two time points
sample_time_interval = 1.0290120765754769
x_acc = (x_speed.diff() / sample_time_interval).fillna(0)
y_acc = (y_speed.diff() / sample_time_interval).fillna(0)
# Limit the acceleration range to -2 ~ 2
x_acc = np.clip(x_acc, -5, 5)
y_acc = np.clip(y_acc, -5, 5)
# 4.Plot the acceleration in the x- and y-axis directions
df_gps['x_acc'] = x_acc
df_gps['y_acc'] = y_acc
df_gps.rename(columns={'seconds_elapsed': 'Time(s)'}, inplace=True)
plot_df(df_gps[['Time(s)', 'x_acc', 'y_acc']], 0, 7000, 'GPS x and y acceleration', 'gps_x_y_acceleration')
# save the data
df_gps.to_csv(os.path.join(current_dir, 'location_offset_with_x_y_acceleration.csv'), index=False)

