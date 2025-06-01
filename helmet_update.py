#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helmet Observation Behavior Analysis System - Complete Version
Used to analyze cycling helmet IMU data, identify observation behavior and bicycle motion status

Main functions:
- Load and preprocess helmet IMU data
- Generate features for behavior recognition
- Mark observation behavior (forward, left, right, etc.)
- Train multiple machine learning models for behavior recognition
- Generate visualization results and analysis reports
- Support integration with bicycle data
- Support GPS trajectory analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy import signal
from scipy.stats import entropy
from scipy.spatial.transform import Rotation as R
import time
import joblib
from matplotlib.font_manager import FontProperties
import os
import folium
from folium.plugins import HeatMap
import logging
import argparse
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, TextIO
import plotly.express as px
import plotly.graph_objects as go
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import json
warnings.filterwarnings("ignore")


# Configuring the logging system
def setup_logger(log_file=None, log_level=logging.INFO) -> logging.Logger:
    """Setting up the logger"""
    logger = logging.getLogger("helmet_analysis")
    logger.setLevel(log_level)

    # Clearing existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# Creating a default logger
logger = setup_logger()

# Default Configuration
DEFAULT_CONFIG = {
    'window_size': 80,  # Sliding window size
    'head_turn_threshold': 0.8,  # Head rotation angular velocity threshold
    'mirror_check_threshold': 0.6,  # Mirror check angular velocity threshold
    'expansion_factor': 0.05,  # Location Boundary Expansion Factor
    'sampling_rate': 10,  # Data sampling rate (Hz)
    'frequency_segment_size': 20,  # Frequency domain analysis segment size
    'random_state': 42,  # Random Seed
}


class HelmetAnalyzer:
    """
    Helmet observation behavior analysis system main class
    Processes helmet IMU data and identifies observation behavior
    """

    def __init__(self, config=None):
        """
        Initializing the Analyzer

        Parameters:
            config (dict, optional): Analyzer configuration, if not provided the default configuration is used
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.logger = logger  # Using the global logger

        # Set up matplotlib configuration to support Chinese and English
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = True

        # Initialize the model and data container
        self.helmet_data = None
        self.bike_data = None
        self.labeled_data = None
        self.models = {}
        self.results = {}

    def load_helmet_data(self, file_path: str) -> pd.DataFrame:
        """
        Loading and preprocessing helmet IMU data

        Parameters:
            file_path (str):Excel file path

        Returns:
            pd.DataFrame: Preprocessed data, returns None if failed

        Raises:
            FileNotFoundError: File does not exist
            ValueError: Data does not conform to the expected format
        """
        try:
            self.logger.info(f"Loading helmet data: {file_path}")

            # Check if a file exists
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")

            # Loading data
            df = pd.read_excel(file_path)

            # Display basic information
            self.logger.info(f"Data shape: {df.shape}")
            self.logger.info(f"Column Name: {df.columns.tolist()}")

            # Checking for required columns
            required_imu_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
            missing_columns = [col for col in required_imu_columns if col not in df.columns]

            if missing_columns:
                self.logger.warning(f"Recommended IMU column missing: {missing_columns}")
                if all(col not in df.columns for col in required_imu_columns):
                    raise ValueError("Data is missing all required IMU columns (ax, ay, az, wx, wy, wz)")

            # Checking and handling missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                self.logger.info(f"Finding missing values: {missing_values[missing_values > 0]}")
                self.logger.info("Filling missing values using time series methods...")
                df = df.ffill().bfill()

            # Replace infinite values with NaN, then fill
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill()

            # Make sure you have a timestamp column
            if 'timestamp' not in df.columns and 'Time(s)' in df.columns:
                df['timestamp'] = df['Time(s)']

            self.helmet_data = df
            return df

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def create_helmet_features(self, df: pd.DataFrame = None, window_size: int = None) -> pd.DataFrame:
        """
        Create features from helmet IMU data to identify observed behaviors

        Parameters:
            df (pd.DataFrame, optional): Raw helmet IMU data, if not provided, the loaded data is used
            window_size (int, optional): Sliding window size, if not provided the config value is used

        Returns:
            pd.DataFrame: Data with features
        """
        self.logger.info("Creating a Helmet Observation Behavior Signature...")

        if df is None:
            if self.helmet_data is None:
                raise ValueError("No data provided and no helmet data preloaded")
            df = self.helmet_data

        # Use the window size from the configuration, or use the provided value
        window_size = window_size or self.config['window_size']

        # Copy data to prevent modification of the original data
        data = df.copy()

        # Record the original number of columns for calculating the number of features
        original_column_count = len(df.columns)

        try:
            # Calculate the helmet position and orientation features
            if all(col in data.columns for col in ['ax', 'ay', 'az']):
                # Linear acceleration characteristics
                self._create_acceleration_features(data, window_size)

            # Angular velocity characteristics
            if all(col in data.columns for col in ['wx', 'wy', 'wz']):
                # Total amount and characteristics of angular velocity
                self._create_angular_velocity_features(data, window_size)

            # Quaternion orientation feature (if present)
            self._create_quaternion_features(data, window_size)

            # Frequency domain features (using angular velocity data)
            if all(col in data.columns for col in ['wx', 'wy', 'wz']):
                self._create_frequency_domain_features(data)

            # Calculate the head rotation direction duration feature
            if all(col in data.columns for col in ['wx', 'wy', 'wz']):
                self._create_head_rotation_features(data)

            # Ensure that there are no infinite values or  NaN values
            data = data.replace([np.inf, -np.inf], np.nan)

            # Checking for NaN Values
            nan_cols = data.columns[data.isna().any()].tolist()
            if nan_cols:
                self.logger.warning(f"Warning: Found columns with NaN values: {nan_cols}")
                self.logger.info("Fill NaN values with column mean...")

                # Fill NaN values with the mean for each column
                for col in nan_cols:
                    col_mean = data[col].mean()
                    if np.isnan(col_mean):
                        data[col] = 0
                    else:
                        data[col] = data[col].fillna(col_mean)

            # Calculating the number of features
            feature_count = len(data.columns) - original_column_count
            self.logger.info(f"created {feature_count} features")

            return data

        except Exception as e:
            self.logger.error(f"Error creating feature: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise

    def _create_acceleration_features(self, data: pd.DataFrame, window_size: int):
        """Creating Acceleration-Related Features"""
        # Calculate linear acceleration characteristics
        data['total_acc'] = np.sqrt(data['ax'] ** 2 + data['ay'] ** 2 + data['az'] ** 2)
        data['acc_xy'] = np.sqrt(data['ax'] ** 2 + data['ay'] ** 2)  # Horizontal acceleration
        data['acc_ratio_z'] = np.abs(data['az']) / (data['total_acc'] + 1e-10)  # z-axis acceleration ratio

        # Creating a acceleration difference change
        for axis in ['ax', 'ay', 'az']:
            data[f'{axis}_diff'] = data[axis].diff().fillna(0)
        data['total_acc_diff'] = data['total_acc'].diff().fillna(0)

        # Rolling Window Features
        for axis in ['ax', 'ay', 'az', 'total_acc']:
            # Rolling mean, standard deviation, maximum, minimum
            data[f'{axis}_mean'] = data[axis].rolling(window=window_size, center=True).mean()
            data[f'{axis}_std'] = data[axis].rolling(window=window_size, center=True).std()
            data[f'{axis}_max'] = data[axis].rolling(window=window_size, center=True).max()
            data[f'{axis}_min'] = data[axis].rolling(window=window_size, center=True).min()

            # Filling NaN values
            data[f'{axis}_mean'] = data[f'{axis}_mean'].ffill().bfill()
            data[f'{axis}_std'] = data[f'{axis}_std'].ffill().bfill()
            data[f'{axis}_max'] = data[f'{axis}_max'].ffill().bfill()
            data[f'{axis}_min'] = data[f'{axis}_min'].ffill().bfill()

            # Calculation range (maximum value - minimum value)
            data[f'{axis}_range'] = data[f'{axis}_max'] - data[f'{axis}_min']

    def _create_angular_velocity_features(self, data: pd.DataFrame, window_size: int):
        """Creating Angular Velocity-Related Features"""
        # Calculate the total angular velocity
        data['total_angular_vel'] = np.sqrt(data['wx'] ** 2 + data['wy'] ** 2 + data['wz'] ** 2)
        data['angular_vel_xy'] = np.sqrt(data['wx'] ** 2 + data['wy'] ** 2)  # Horizontal angular velocity
        data['angular_ratio_z'] = np.abs(data['wz']) / (data['total_angular_vel'] + 1e-10)  # z-axis angular velocity ratio

        # Creating the Rate of Change of Angular Velocity
        for axis in ['wx', 'wy', 'wz']:
            data[f'{axis}_diff'] = data[axis].diff().fillna(0)
        data['total_angular_vel_diff'] = data['total_angular_vel'].diff().fillna(0)

        # Angular velocity rolling window feature
        for axis in ['wx', 'wy', 'wz', 'total_angular_vel']:
            data[f'{axis}_mean'] = data[axis].rolling(window=window_size, center=True).mean()
            data[f'{axis}_std'] = data[axis].rolling(window=window_size, center=True).std()
            data[f'{axis}_max'] = data[axis].rolling(window=window_size, center=True).max()
            data[f'{axis}_min'] = data[axis].rolling(window=window_size, center=True).min()

            # Filling NaN values
            data[f'{axis}_mean'] = data[f'{axis}_mean'].ffill().bfill()
            data[f'{axis}_std'] = data[f'{axis}_std'].ffill().bfill()
            data[f'{axis}_max'] = data[f'{axis}_max'].ffill().bfill()
            data[f'{axis}_min'] = data[f'{axis}_min'].ffill().bfill()

            data[f'{axis}_range'] = data[f'{axis}_max'] - data[f'{axis}_min']

    def _create_quaternion_features(self, data: pd.DataFrame, window_size: int):
        """Create quaternion orientation related features (if any))"""
        quat_cols = [col for col in data.columns if col.startswith('q')]
        if len(quat_cols) == 4:

            for quat in quat_cols:
                data[f'{quat}_diff'] = data[quat].diff().fillna(0)

            #
            for quat in quat_cols:
                data[f'{quat}_mean'] = data[quat].rolling(window=window_size, center=True).mean()
                data[f'{quat}_std'] = data[quat].rolling(window=window_size, center=True).std()

                # Filling NaN values
                data[f'{quat}_mean'] = data[f'{quat}_mean'].ffill().bfill()
                data[f'{quat}_std'] = data[f'{quat}_std'].ffill().bfill()

    def _create_frequency_domain_features(self, data: pd.DataFrame):
        """Creating frequency domain features (using angular velocity data)"""
        segment_size = self.config['frequency_segment_size']  # Segment size for frequency domain analysis
        sampling_rate = self.config['sampling_rate']  # sampling fre
        n_segments = len(data) // segment_size

        # Preallocate arrays for efficiency
        for axis in ['wx', 'wy', 'wz']:
            # Main frequency
            data[f'{axis}_dom_freq'] = np.zeros(len(data))
            # Spectrum Energy
            data[f'{axis}_spectral_energy'] = np.zeros(len(data))
            # Spectral entropy
            data[f'{axis}_spectral_entropy'] = np.zeros(len(data))

        # Using parallel processing to accelerate frequency domain feature computation
        with ThreadPoolExecutor(max_workers=min(8, cpu_count())) as executor:
            futures = []
            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = min(start_idx + segment_size, len(data))
                futures.append(executor.submit(
                    self._calculate_frequency_features,
                    data, start_idx, end_idx, sampling_rate
                ))

            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error computing frequency domain features: {e}")

    def _calculate_frequency_features(self, data, start_idx, end_idx, sampling_rate):
        """Compute frequency domain features for a given data segment"""
        for axis in ['wx', 'wy', 'wz']:
            signal_data = data[axis].iloc[start_idx:end_idx].values

            # Calculate power spectral density
            try:
                f, Pxx = signal.periodogram(signal_data, fs=sampling_rate)

                # Finding the main frequency
                if len(f) > 1 and np.sum(Pxx) > 0:
                    dom_freq_idx = np.argmax(Pxx[1:]) + 1
                    dom_freq = f[dom_freq_idx]

                    # Calculate spectral energy
                    spectral_energy = np.sum(Pxx)

                    # Calculate spectral entropy
                    psd_norm = Pxx / np.sum(Pxx)
                    # Handling zero values to avoid log(0)
                    psd_norm = np.where(psd_norm > 0, psd_norm, 1e-10)
                    spectral_entropy_val = entropy(psd_norm)

                    # Store the frequency domain characteristics of this segment
                    data.loc[start_idx:end_idx - 1, f'{axis}_dom_freq'] = dom_freq
                    data.loc[start_idx:end_idx - 1, f'{axis}_spectral_energy'] = spectral_energy
                    data.loc[start_idx:end_idx - 1, f'{axis}_spectral_entropy'] = spectral_entropy_val
            except Exception as e:
                self.logger.error(f"calculate axis {axis} Error in frequency domain features: {e}")
                # Filling in default values
                data.loc[start_idx:end_idx - 1, f'{axis}_dom_freq'] = 0
                data.loc[start_idx:end_idx - 1, f'{axis}_spectral_energy'] = 0
                data.loc[start_idx:end_idx - 1, f'{axis}_spectral_entropy'] = 0

    def _create_head_rotation_features(self, data: pd.DataFrame):
        """Calculate the head rotation direction duration feature"""
        # Define horizontal rotation threshold
        horiz_rotation_threshold = self.config['head_turn_threshold'] * 0.5  # Adjust the threshold to be below the observed behavior threshold

        # Initializing variables
        rotating_left = data['wz'] > horiz_rotation_threshold
        rotating_right = data['wz'] < -horiz_rotation_threshold

        # Calculate the duration of the continuous left and right turns
        # # This part is difficult to vectorize, but the loop logic can be optimized
        data['left_turn_duration'] = 0
        data['right_turn_duration'] = 0

        # Using numpy arrays for more efficient processing
        wz_values = data['wz'].values
        left_durations = np.zeros(len(wz_values), dtype=np.int32)
        right_durations = np.zeros(len(wz_values), dtype=np.int32)

        current_left_duration = 0
        current_right_duration = 0

        for i in range(len(wz_values)):
            if wz_values[i] > horiz_rotation_threshold:  # left turn
                current_left_duration += 1
                current_right_duration = 0
            elif wz_values[i] < -horiz_rotation_threshold:  # right turn
                current_right_duration += 1
                current_left_duration = 0
            else:  # no turning
                current_left_duration = 0
                current_right_duration = 0

            left_durations[i] = current_left_duration
            right_durations[i] = current_right_duration

        # Assign the result to a DataFrame
        data['left_turn_duration'] = left_durations
        data['right_turn_duration'] = right_durations

    def integrate_with_bike_data(self, helmet_df: pd.DataFrame = None, bike_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Integrate helmet IMU data with bicycle IMU data for normalization

        Parameters:
            helmet_df (pd.DataFrame, optional): Helmet IMU data, if not provided, use the loaded data
            bike_df (pd.DataFrame, optional): bike IMU data, if not provided, use the loaded data

        Returns:
            pd.DataFrame: Integrated data
        """
        self.logger.info("Integrate helmet and bike data...")


        if helmet_df is None:
            if self.helmet_data is None:
                raise ValueError("No helmet data provided and not preloaded")
            helmet_df = self.helmet_data

        if bike_df is None:
            if self.bike_data is None:
                self.logger.info("No bike data provided and not preloaded")
                return helmet_df.copy()
            bike_df = self.bike_data

        #
        data = helmet_df.copy()

        try:
            # Make sure both datasets have the same timeline
            if 'timestamp' in helmet_df.columns and 'timestamp' in bike_df.columns:
                self.logger.info("Integrate data based on timestamp...")

                # Extract the bicycle acceleration and angular velocity columns
                bike_acc_cols = [col for col in bike_df.columns if col.startswith('a') and len(col) == 2]
                bike_gyro_cols = [col for col in bike_df.columns if col.startswith('w') and len(col) == 2]

                # Rename the bicycle column to avoid conflicts
                bike_columns = {}
                for col in bike_acc_cols + bike_gyro_cols:
                    bike_columns[col] = f'bike_{col}'

                bike_subset = bike_df[['timestamp'] + bike_acc_cols + bike_gyro_cols].rename(columns=bike_columns)

                # Merge datasets and sort by timestamp
                merged_data = pd.merge_asof(
                    helmet_df.sort_values('timestamp'),
                    bike_subset.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest'
                )

                if 'timestamp' in helmet_df.columns and 'timestamp' in bike_df.columns:
                    # Checking the quality of time synchronization
                    time_diff = merged_data['timestamp'].diff().fillna(0)
                    irregular_intervals = (time_diff > time_diff.median() * 2).sum()

                    if irregular_intervals > len(merged_data) * 0.1:  # If more than 10% of the intervals are irregular
                        self.logger.warning(f"detect {irregular_intervals} irregular time intervals, there may be problems with data synchronization")
                        # choose interpolation or other processing methods
                        merged_data = merged_data.interpolate(method='time')

                # Calculate the acceleration and angular velocity of the helmet relative to the bicycle
                merged_data = self._calculate_relative_motion(merged_data)

                return merged_data
            else:
                # Try an alternative approach - assuming the datasets have the same number of rows
                if len(helmet_df) == len(bike_df):
                    self.logger.warning("The timestamp is missing in the dataset, but the frame length is equal, try to merge directly...")

                    # Extract the bicycle acceleration and angular velocity columns
                    bike_acc_cols = [col for col in bike_df.columns if col.startswith('a') and len(col) == 2]
                    bike_gyro_cols = [col for col in bike_df.columns if col.startswith('w') and len(col) == 2]

                    # Rename the bicycle column to avoid conflicts
                    bike_columns = {}
                    for col in bike_acc_cols + bike_gyro_cols:
                        bike_columns[col] = f'bike_{col}'

                    # Add bicycle data directly to helmet data
                    for col in bike_acc_cols + bike_gyro_cols:
                        data[bike_columns[col]] = bike_df[col].values

                    # Calculate the acceleration and angular velocity of the helmet relative to the bicycle
                    data = self._calculate_relative_motion(data)

                    return data
                else:
                    self.logger.warning("The data sets are missing timestamps and have different frame lengths, making it impossible to integrate the data")
                    return data
        except Exception as e:
            self.logger.error(f"Error integrating data: {str(e)}")
            self.logger.info("Will continue to use only helmet data for analysis")
            return data

    def _calculate_relative_motion(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculating the Motion of the Helmet Relative to the Bicycle """
        # Calculate the acceleration and angular velocity of the helmet relative to the bicycle
        for axis in ['x', 'y', 'z']:
            if f'a{axis}' in data.columns and f'bike_a{axis}' in data.columns:
                # Calculating relative acceleration
                data[f'rel_a{axis}'] = data[f'a{axis}'] - data[f'bike_a{axis}']

                # Calculate the absolute value of relative acceleration - more concerned with relative motion intensity
                data[f'rel_a{axis}_abs'] = np.abs(data[f'rel_a{axis}'])

            if f'w{axis}' in data.columns and f'bike_w{axis}' in data.columns:
                # Calculate relative angular velocity
                data[f'rel_w{axis}'] = data[f'w{axis}'] - data[f'bike_w{axis}']

                # Calculate the absolute value of the relative angular velocity
                data[f'rel_w{axis}_abs'] = np.abs(data[f'rel_w{axis}'])

        # Calculate the total relative motion characteristics
        if all(f'rel_a{axis}' in data.columns for axis in ['x', 'y', 'z']):
            data['rel_total_acc'] = np.sqrt(
                data['rel_ax'] ** 2 + data['rel_ay'] ** 2 + data['rel_az'] ** 2
            )
            # Add the component of relative acceleration in the horizontal plane - more important for observing behavior
            data['rel_acc_horizontal'] = np.sqrt(data['rel_ax'] ** 2 + data['rel_ay'] ** 2)

        if all(f'rel_w{axis}' in data.columns for axis in ['x', 'y', 'z']):
            data['rel_total_angular_vel'] = np.sqrt(
                data['rel_wx'] ** 2 + data['rel_wy'] ** 2 + data['rel_wz'] ** 2
            )
            # Focus on the z-axis relative angular velocity - this is most important for head turning
            data['rel_wz_emphasis'] = data['rel_wz'] * 2.0  # Weighted Emphasis

        # Adding sport consistency features - relevance of helmets and cycling
        for axis in ['x', 'y', 'z']:
            if f'a{axis}' in data.columns and f'bike_a{axis}' in data.columns:
                # Compute motion coherence (approximation of correlation)
                data[f'motion_consistency_a{axis}'] = data[f'a{axis}'] * data[f'bike_a{axis}']

            if f'w{axis}' in data.columns and f'bike_w{axis}' in data.columns:
                data[f'motion_consistency_w{axis}'] = data[f'w{axis}'] * data[f'bike_w{axis}']

        return data

    def label_observation_behaviors(self, df: pd.DataFrame = None,
                                    head_turn_threshold: float = None,
                                    mirror_check_threshold: float = None) -> pd.DataFrame:
        """
        Labeling helmet observation behavior based on IMU data

        Parameters:
            df (pd.DataFrame, optional): Helmet data with features
            head_turn_threshold (float, optional): Head rotation angular velocity threshold
            mirror_check_threshold (float, optional): Mirror check angular velocity threshold

        Returns:
            pd.DataFrame: Data with observed behavior labels
        """
        self.logger.info("Tagging Helmet Observation Behavior...")

        if df is None:
            if self.helmet_data is None:
                raise ValueError("No data provided and no data preloaded")
            df = self.helmet_data

        # Use the threshold in the configuration or the provided value
        head_turn_threshold = head_turn_threshold or self.config['head_turn_threshold']
        mirror_check_threshold = mirror_check_threshold or self.config['mirror_check_threshold']

        # Copy data to prevent modification of the original data
        data = df.copy()

        # Initialize observation behavior label
        data['observation'] = 'forward'  # The default is forward-looking.

        # Different observation behaviors are labeled based on angular velocity and duration
        if all(col in data.columns for col in ['wx', 'wy', 'wz', 'left_turn_duration', 'right_turn_duration']):
            # Define the observation behavior conditions

            # Left side observation condition
            left_obs = (data['wz'] > head_turn_threshold) & (data['left_turn_duration'] >= 3)

            # Right side viewing conditions
            right_obs = (data['wz'] < -head_turn_threshold) & (data['right_turn_duration'] >= 3)

            # Mirror check condition (usually a slight head turn)
            left_mirror = ((data['wz'] > mirror_check_threshold) &
                           (data['wz'] <= head_turn_threshold) &
                           (data['left_turn_duration'] >= 2))

            right_mirror = ((data['wz'] < -mirror_check_threshold) &
                            (data['wz'] >= -head_turn_threshold) &
                            (data['right_turn_duration'] >= 2))

            # Head-up condition (check far away)
            look_up = (data['wx'] < -head_turn_threshold)

            # Head down condition (check nearby)
            look_down = (data['wx'] > head_turn_threshold)

            # Life saver observation (strong head turning movement)
            lifesaver_left = (data['wz'] > head_turn_threshold * 1.5) & (data['left_turn_duration'] >= 5)
            lifesaver_right = (data['wz'] < -head_turn_threshold * 1.5) & (data['right_turn_duration'] >= 5)

            # Marking Observation Behavior
            data.loc[left_obs, 'observation'] = 'left_side'
            data.loc[right_obs, 'observation'] = 'right_side'
            data.loc[left_mirror, 'observation'] = 'left_mirror'
            data.loc[right_mirror, 'observation'] = 'right_mirror'
            data.loc[look_up, 'observation'] = 'far'
            data.loc[look_down, 'observation'] = 'near'
            data.loc[lifesaver_left, 'observation'] = 'lifesaver_left'
            data.loc[lifesaver_right, 'observation'] = 'lifesaver_right'

            # Intermediate observations are more difficult to accurately identify, and simplified conditions are used here
            # # (head slightly raised but not reaching the "far" level)
            mid_level = (~look_up & ~look_down & ~left_obs & ~right_obs &
                         ~left_mirror & ~right_mirror & ~lifesaver_left & ~lifesaver_right)
            data.loc[mid_level, 'observation'] = 'mid_level'


            # save the data
            self.labeled_data = data

            return data
        else:
            self.logger.warning("The data lacks the necessary angular velocity or duration columns to label the observed behavior")
            return data


    def train_observation_models(self, df: pd.DataFrame = None,
                                 use_bike_data: bool = False,
                                 output_dir: str = 'results') -> Tuple[Dict, Dict]:
        """
        Training three different models to recognize helmet viewing behavior

        Parameters:
            df (pd.DataFrame, optional): Data with features and labels
            use_bike_data (bool): Whether to use bicycle data
            output_dir (str): Directory to save the results

        Returns:
            tuple: (Training model dictionary, performance result dictionary)
        """
        self.logger.info("\nStart helmet observation behavior recognition model training...")

        if df is None:
            if self.labeled_data is None:
                raise ValueError("No data provided and no pre-labeled data")
            df = self.labeled_data

        # Create Output Directory
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Select feature columns
            feature_cols = self._select_observation_features(df, use_bike_data)

            # Prepare feature matrix and target vector
            X = df[feature_cols]
            y = df['observation']

            # Using the Tag Encoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Check for NaN values in X
            if X.isna().any().any():
                nan_count = X.isna().sum().sum()
                self.logger.warning(f"warning: The feature matrix has {nan_count} NaN values will be processed in the pipeline")

            # Divide into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=self.config['random_state'], stratify=y_encoded
            )

            # Define three different models
            models = self._define_classification_models()

            # Creating a pipeline with preprocessing
            pipelines = {}
            results = {}

            # Train and evaluate each model
            for name, model in models.items():
                self.logger.info(f"\ntraining {name} model...")

                # Creating a pipeline - Adding a feature selection step
                if use_bike_data:
                    # When using bicycle data, add feature selection to remove irrelevant features
                    pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler()),
                        ('feature_selector', SelectFromModel(
                            RandomForestClassifier(n_estimators=50, random_state=self.config['random_state']),
                            threshold='0.75*mean'  # **Increase the threshold to keep only the most important features**
                        )),
                        ('pca', PCA(n_components=0.95)),  # **Reducing PCA components**
                        ('classifier', model)
                    ])
                else:
                    # Original pipeline without bicycle data
                    pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=0.95)),
                        ('classifier', model)
                    ])

                # Perform cross validation
                self.logger.info("Perform 5-fold cross validation...")
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
                self.logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                # Recording start time
                start_time = time.time()

                # Training the model
                pipeline.fit(X_train, y_train)

                # Record training time
                train_time = time.time() - start_time

                # Evaluate on the test set
                y_pred = pipeline.predict(X_test)

                # Recording forecast time
                start_time = time.time()
                _ = pipeline.predict(X_test[:100])  # Predict 100 samples to measure performance
                predict_time = (time.time() - start_time) / 100  # Average prediction time per sample

                # Convert the encoded labels back to the original categories
                y_test_original = label_encoder.inverse_transform(y_test)
                y_pred_original = label_encoder.inverse_transform(y_pred)

                # Calculating evaluation metrics
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test_original, y_pred_original, output_dict=True)

                # Calculate the confusion matrix
                cm = confusion_matrix(y_test, y_pred)

                # Storing Results
                results[name] = {
                    'accuracy': accuracy,
                    'cv_scores': cv_scores,
                    'classification_report': report,
                    'confusion_matrix': cm,
                    'train_time': train_time,
                    'predict_time_per_sample': predict_time,
                    'feature_importances': None  # Will populate the Random Forest model
                }

                # For Random Forest, save feature importance
                if name == "Random Forest":
                    rf_model = pipeline.named_steps['classifier']

                    # Try to get feature importance
                    try:
                        # Get the feature importance after PCA transformation
                        pca = pipeline.named_steps['pca']
                        pca_features = [f'PC{i + 1}' for i in range(pca.n_components_)]

                        importances_df = pd.DataFrame({
                            'Feature': pca_features,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=False)

                        results[name]['feature_importances'] = importances_df

                        # Try mapping back to the original features
                        components_df = pd.DataFrame(
                            pca.components_,
                            columns=feature_cols,
                            index=pca_features
                        )
                        results[name]['pca_components'] = components_df
                    except Exception as e:
                        self.logger.error(f"Error getting feature importance: {e}")

                self.logger.info(f"{name} accuracy: {accuracy:.4f}")
                self.logger.info(f"training time: {train_time:.2f} seconds")
                self.logger.info(f"Prediction time for each sample: {predict_time * 1000:.4f} milliseconds")

                pipelines[name] = pipeline

            # Saving the model and results
            self.observation_models = pipelines
            self.observation_results = results

            # Analysis results - save graphs to output_dir
            self._plot_observation_model_results(results, label_encoder.classes_, output_dir)

            # Generate confusion matrix visualization - save to output_dir
            self._plot_confusion_matrices(results, label_encoder.classes_, output_dir=output_dir,
                                          prefix="observation_")


            return pipelines, results

        except Exception as e:
            self.logger.error(f"Error while training model: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise

    def _select_observation_features(self, df: pd.DataFrame, use_bike_data: bool) -> List[str]:
        """Selecting feature columns for training"""
        feature_cols = []

        # Helmet IMU characteristics (remain unchanged)
        imu_prefixes = ['a', 'w', 'total_acc', 'total_angular_vel']
        imu_suffixes = ['', '_diff', '_mean', '_std', '_max', '_min', '_range']

        for prefix in imu_prefixes:
            for axis in ['x', 'y', 'z']:
                for suffix in imu_suffixes:
                    col = f'{prefix}{axis}{suffix}'
                    if col in df.columns:
                        feature_cols.append(col)

        # Adding other important features
        for col in df.columns:
            if (col.startswith('q') and '_' in col) or \
                    col.endswith('_duration') or \
                    '_dom_freq' in col or \
                    '_spectral_' in col:
                feature_cols.append(col)

        # If using bicycle data, only add the most core relative features**
        if use_bike_data:
            # **Only the most important relative angular velocity features are retained**
            priority_relative_features = [
                'rel_wz',  # Most important: relative z-axis angular velocity
                'rel_total_angular_vel',  # Total relative angular velocity
                'rel_acc_horizontal',  # Horizontal relative acceleration
            ]

            for col in priority_relative_features:
                if col in df.columns:
                    feature_cols.append(col)

        # Keep only the columns that are actually in the final feature list
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.logger.info(f"use {len(feature_cols)} features for training")

        return feature_cols


    def _define_classification_models(self) -> Dict[str, Any]:
        """Define the classification model for training"""
        random_state = self.config['random_state']

        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state,
                class_weight='balanced'
            ),
            "SVM": SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=random_state
            ),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=500,
                random_state=random_state
            )
        }

        return models

    def _plot_observation_model_results(self, results: Dict, class_names: List, output_dir: str):
        """
        Plotting the performance results of the observed behavior recognition model

        Parameters:
            results (dict): Model evaluation results
            class_names (list): Class Name
            output_dir (str): Directory to save the results
        """
        try:
            # Accuracy comparison
            plt.figure(figsize=(10, 6))
            model_names = list(results.keys())
            accuracies = [results[name]['accuracy'] for name in model_names]

            bar_plot = plt.bar(model_names, accuracies, color=['blue', 'green', 'red'])

            # Display accuracy value on each bar
            for bar, acc in zip(bar_plot, accuracies):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{acc:.4f}', ha='center', fontsize=11)

            plt.title('Comparison of Model Accuracy(helmet)', fontsize=14)
            plt.xlabel('model', fontsize=12)
            plt.ylabel('accuracy', fontsize=12)
            plt.ylim(0, 1.0)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            # save
            plt.savefig(os.path.join(output_dir, 'observation_model_accuracy.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Time performance comparison
            plt.figure(figsize=(12, 6))
            train_times = [results[name]['train_time'] for name in model_names]
            predict_times = [results[name]['predict_time_per_sample'] * 1000 for name in
                             model_names]  # Convert to milliseconds

            x = np.arange(len(model_names))
            width = 0.35

            plt.bar(x - width / 2, train_times, width, label='training time(s)', color='blue')
            plt.bar(x + width / 2, predict_times, width, label='Prediction time per sample(ms)', color='orange')

            plt.title('Comparison of Model Time Performance', fontsize=14)
            plt.xlabel('model', fontsize=12)
            plt.xticks(x, model_names)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            # save
            plt.savefig(os.path.join(output_dir, 'observation_model_time.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Comparison of F1 scores for each class
            plt.figure(figsize=(12, 8))

            # Calculate the F1 score for each class
            classes_f1 = {}
            for cls in class_names:
                classes_f1[cls] = []
                for name in model_names:
                    report = results[name]['classification_report']
                    if cls in report:
                        classes_f1[cls].append(report[cls]['f1-score'])
                    else:
                        classes_f1[cls].append(0)

            # Draw multiple groups of bar charts
            x = np.arange(len(class_names))
            width = 0.25

            for i, (name, color) in enumerate(zip(model_names, ['blue', 'green', 'red'])):
                f1_scores = [classes_f1[cls][i] for cls in class_names]
                plt.bar(x + (i - 1) * width, f1_scores, width, label=name, color=color, alpha=0.7)

            plt.title('F1 score for each observed behavior', fontsize=14)
            plt.xlabel('observed behavior', fontsize=12)
            plt.ylabel('F1-score', fontsize=12)
            plt.xticks(x, class_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            # save
            plt.savefig(os.path.join(output_dir, 'observation_f1_scores.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Create a cross-validation accuracy comparison chart
            plt.figure(figsize=(10, 6))
            cv_means = [results[name]['cv_scores'].mean() for name in model_names]
            cv_stds = [results[name]['cv_scores'].std() for name in model_names]

            bar_plot = plt.bar(model_names, cv_means, yerr=cv_stds, capsize=10, color=['blue', 'green', 'red'])

            # Show cross validation accuracy on each bar
            for bar, mean in zip(bar_plot, cv_means):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{mean:.4f}', ha='center', fontsize=11)

            plt.title('Comparison of Cross Validation Accuracy', fontsize=14)
            plt.xlabel('model', fontsize=12)
            plt.ylabel('Cross validation accuracy', fontsize=12)
            plt.ylim(0, 1.0)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            # save
            plt.savefig(os.path.join(output_dir, 'observation_model_cv_accuracy.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.error(f"error if plotting observation model results: {e}")

    def _plot_confusion_matrices(self, results: Dict, class_names: List, output_dir: str = 'results', prefix: str = ''):
        """
        Generate and save confusion matrix visualizations for each model

        Parameters:
            results (dict): Model evaluation results including classification reports
            class_names (list): Class name list
            output_dir (str): Directory where the charts are saved
            prefix (str): Output file name prefix
        """
        self.logger.info("\nGenerate confusion matrix visualization...")

        # Check if the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Create Output Directory: {output_dir}")

        # For each model
        for model_name, result in results.items():
            # Get the confusion matrix from the results
            if 'confusion_matrix' in result:
                cm = result['confusion_matrix']

                # Create larger graphics for better readability
                plt.figure(figsize=(12, 10))

                # Normalize the confusion matrix to show percentages
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                # Creating more visually appealing heatmaps
                ax = plt.subplot()

                # Use better color-diverging colormaps for visualization
                cmap = sns.color_palette("Blues", as_cmap=True)

                # Plot a heatmap containing raw counts and percentages
                heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                                      xticklabels=class_names, yticklabels=class_names,
                                      cbar=True, square=True, linewidths=.5, ax=ax)

                # Set Chinese and English labels and improve styles
                plt.title(f'confusion matrix - {model_name}', fontsize=16, fontweight='bold')
                plt.xlabel('Prediction label', fontsize=14)
                plt.ylabel('Real label', fontsize=14)

                # Improved tick label sizing
                plt.xticks(fontsize=10, rotation=45, ha='right')
                plt.yticks(fontsize=10)

                # Add a grid for better readability
                ax.set_axisbelow(True)

                # Add a text note with percentage precision in each cell
                thresh = cm.max() / 2
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        # Add percentage text below the count
                        ax.text(j + 0.5, i + 0.7, f'{cm_norm[i, j]:.1%}',
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black",
                                fontsize=9)

                # Set up a compact layout and save
                plt.tight_layout()

                # Save the image to the output directory with high resolution
                filename = os.path.join(output_dir, f"{prefix}{model_name.replace(' ', '_')}_confusion_matrix.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()

                self.logger.info(f"Confusion matrix saved: {filename}")
            else:
                self.logger.warning(f"warning: cannot found {model_name} confusion matrix")



    def train_bike_motion_from_helmet(self, df: pd.DataFrame = None,
                                      bike_labels: pd.DataFrame = None,
                                      output_dir: str = 'results') -> Tuple[Dict, Dict]:
        """
        Training a model to identify cycling motion from helmet IMU data

        Parameters:
            df (pd.DataFrame, optional): helmet IMU data
            bike_labels (pd.DataFrame, optional): Cycling label, optional
            output_dir (str): save

        Returns:
            tuple: (Training model dictionary, performance result dictionary)
        """
        self.logger.info("\nStart identifying cycling status from helmet data...")

        if df is None:
            if self.labeled_data is None:
                raise ValueError("No data provided and no pre-labeled data")
            df = self.labeled_data

        # save
        os.makedirs(output_dir, exist_ok=True)

        try:
            # If no bicycle label is provided, create a mock label for demonstration
            if bike_labels is None:
                self.logger.info("If no bicycle label is provided, create a mock label for demonstration")
                # Simple inference based on the angular velocity and acceleration of the helmet
                data = df.copy()

                # Mark straight ahead, speed up, slow down and turn
                data['bike_motion'] = 'straight'  # Default is straight motion

                # Simple heuristic rules
                if 'total_acc' in data.columns and 'total_acc_diff' in data.columns:
                    # Determine acceleration and deceleration based on the total rate of change of acceleration
                    data.loc[data['total_acc_diff'] > 0.1, 'bike_motion'] = 'acceleration'
                    data.loc[data['total_acc_diff'] < -0.1, 'bike_motion'] = 'deceleration'

                if 'wz' in data.columns:
                    # Determine turning based on z-axis angular velocity
                    data.loc[abs(data['wz']) > 0.5, 'bike_motion'] = 'turning'

                bike_motion_labels = data['bike_motion']
            else:
                self.logger.info("Use the provided cycling tags")
                bike_motion_labels = bike_labels['bike_motion']

            # Make sure df has a bike_motion column
            df = df.copy()
            df['bike_motion'] = bike_motion_labels

            # Select feature columns (similar to observed behavior model, but emphasizes acceleration features)
            feature_cols = []

            # Helmet IMU acceleration characteristics
            acc_prefixes = ['a', 'total_acc']
            acc_suffixes = ['', '_diff', '_mean', '_std', '_max', '_min', '_range']

            for prefix in acc_prefixes:
                for axis in ['x', 'y', 'z']:
                    for suffix in acc_suffixes:
                        col = f'{prefix}{axis}{suffix}'
                        if col in df.columns:
                            feature_cols.append(col)

            # Helmet IMU angular velocity characteristics
            gyro_prefixes = ['w', 'total_angular_vel']
            for prefix in gyro_prefixes:
                for axis in ['x', 'y', 'z']:
                    for suffix in acc_suffixes:
                        col = f'{prefix}{axis}{suffix}'
                        if col in df.columns:
                            feature_cols.append(col)

            # Adding frequency domain features
            for col in df.columns:
                if '_dom_freq' in col or '_spectral_' in col:
                    feature_cols.append(col)

            # Only keep the columns that are actually present in the final feature list
            feature_cols = [col for col in feature_cols if col in df.columns]
            self.logger.info(f"use {len(feature_cols)} features to recognize bike motions")

            # Prepare feature matrix and target vector
            X = df[feature_cols]
            y = df['bike_motion']

            # Using the Tag Encoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Check for NaN values in X
            if X.isna().any().any():
                nan_count = X.isna().sum().sum()
                self.logger.warning(f"warning: The feature matrix has {nan_count} NaN values, which will be handled in the pipeline")

            # Divide into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=self.config['random_state'], stratify=y_encoded
            )

            # Define three different models (same as before)
            models = self._define_classification_models()

            # Creating a pipeline with preprocessing
            pipelines = {}
            results = {}

            # Train and evaluate each model
            for name, model in models.items():
                self.logger.info(f"\ntraining {name} Model for bicycle motion recognition...")

                # Create pipeline - add SimpleImputer to handle NaN values
                pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])

                # Perform cross validation
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
                self.logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                #
                start_time = time.time()

                #
                pipeline.fit(X_train, y_train)

                #
                train_time = time.time() - start_time

                # Evaluate on the test set
                y_pred = pipeline.predict(X_test)

                #
                start_time = time.time()
                _ = pipeline.predict(X_test[:100])  # Predict 100 samples to measure performance
                predict_time = (time.time() - start_time) / 100  # Average prediction time per sample

                # Convert the encoded labels back to the original categories
                y_test_original = label_encoder.inverse_transform(y_test)
                y_pred_original = label_encoder.inverse_transform(y_pred)

                #
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test_original, y_pred_original, output_dict=True)

                #
                cm = confusion_matrix(y_test, y_pred)

                # results
                results[name] = {
                    'accuracy': accuracy,
                    'cv_scores': cv_scores,
                    'classification_report': report,
                    'confusion_matrix': cm,
                    'train_time': train_time,
                    'predict_time_per_sample': predict_time
                }

                self.logger.info(f"{name} accuracy: {accuracy:.4f}")
                self.logger.info(f"training time: {train_time:.2f} seconds")
                self.logger.info(f"Prediction time for each sample: {predict_time * 1000:.4f} milliseconds")

                pipelines[name] = pipeline

            #
            self.bike_motion_models = pipelines
            self.bike_motion_results = results

            #
            self._plot_bike_motion_results(results, label_encoder.classes_, output_dir)

            #
            self._plot_confusion_matrices(results, label_encoder.classes_, output_dir=output_dir,
                                          prefix="bike_motion_")



            return pipelines, results

        except Exception as e:
            self.logger.error(f"Error when training the cycling model: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise

    def _plot_bike_motion_results(self, results: Dict, class_names: List, output_dir: str):
        """
        Plotting the performance results of the bicycle motion recognition model

        Parameters:
            results (dict): Model evaluation results
            class_names (list): Class Name
            output_dir (str): Directory to save the results
        """
        try:
            # Accuracy comparison
            plt.figure(figsize=(10, 6))
            model_names = list(results.keys())
            accuracies = [results[name]['accuracy'] for name in model_names]

            bar_plot = plt.bar(model_names, accuracies, color=['blue', 'green', 'red'])

            #
            for bar, acc in zip(bar_plot, accuracies):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{acc:.4f}', ha='center', fontsize=11)

            plt.title('Comparison of Model Accuracy(bicycle)', fontsize=14)
            plt.xlabel('model', fontsize=12)
            plt.ylabel('accuracy', fontsize=12)
            plt.ylim(0, 1.0)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            #
            plt.savefig(os.path.join(output_dir, 'bike_motion_model_accuracy.png'), dpi=300, bbox_inches='tight')
            plt.close()

            #
            plt.figure(figsize=(12, 8))

            #
            classes_f1 = {}
            for cls in class_names:
                classes_f1[cls] = []
                for name in model_names:
                    report = results[name]['classification_report']
                    if cls in report:
                        classes_f1[cls].append(report[cls]['f1-score'])
                    else:
                        classes_f1[cls].append(0)

            #
            x = np.arange(len(class_names))
            width = 0.25

            for i, (name, color) in enumerate(zip(model_names, ['blue', 'green', 'red'])):
                f1_scores = [classes_f1[cls][i] for cls in class_names]
                plt.bar(x + (i - 1) * width, f1_scores, width, label=name, color=color, alpha=0.7)

            plt.title('F1-score of bicycle behavior', fontsize=14)
            plt.xlabel('bicycle behavior', fontsize=12)
            plt.ylabel('F1-score', fontsize=12)
            plt.xticks(x, class_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            # save
            plt.savefig(os.path.join(output_dir, 'bike_motion_f1_scores.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Comparison of cross validation results
            plt.figure(figsize=(10, 6))
            cv_means = [results[name]['cv_scores'].mean() for name in model_names]
            cv_stds = [results[name]['cv_scores'].std() for name in model_names]

            bar_plot = plt.bar(model_names, cv_means, yerr=cv_stds, capsize=10, color=['blue', 'green', 'red'])

            #
            for bar, mean in zip(bar_plot, cv_means):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{mean:.4f}', ha='center', fontsize=11)

            plt.title('Comparison of Cross Validation Accuracy', fontsize=14)
            plt.xlabel('model', fontsize=12)
            plt.ylabel('Cross Validation Accuracy', fontsize=12)
            plt.ylim(0, 1.0)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            # 保存到输出目录
            plt.savefig(os.path.join(output_dir, 'bike_motion_cv_accuracy.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.error(f"Error when plotting cycling model results: {e}")

    def analyze_helmet_imu_importance(self, models_results: Dict = None, output_dir: str = 'results'):
        """
        Analyzing the importance of helmet IMU data in observing behavior and identifying cycling movements

        Parameters:
            models_results (dict, optional): Model training results. If not provided, the stored results are used.
            output_dir (str): Directory to save the results
        """
        self.logger.info("\nAnalyzing the Importance of Helmet IMU Data...")

        if models_results is None:
            if hasattr(self, 'observation_results'):
                models_results = self.observation_results
            else:
                self.logger.error("No model results were provided and no observations were stored")
                return

        #
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Finding feature importance for Random Forest model
            if 'Random Forest' in models_results and 'feature_importances' in models_results['Random Forest']:
                importances_df = models_results['Random Forest']['feature_importances']

                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=importances_df.head(15))
                plt.title('Importance of helmet IMU data features (PCA component)', fontsize=14)
                plt.xlabel('importance', fontsize=12)
                plt.ylabel('feature', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'helmet_feature_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()


                # Analyzing PCA components
                if 'pca_components' in models_results['Random Forest']:
                    components_df = models_results['Random Forest']['pca_components']

                    # Find the original features that contribute most to the 5 most important PCA components
                    top_pca_components = importances_df.head(5)['Feature'].tolist()

                    self.logger.info("\nOriginal features that contribute most to the most important PCA components:")

                    #
                    component_analysis_data = []

                    for pc in top_pca_components:
                        #
                        weights = components_df.loc[pc].abs().sort_values(ascending=False)
                        self.logger.info(f"\n{pc} the top 5 contributing features of:")
                        for i, (feature, weight) in enumerate(weights.head(5).items(), 1):
                            self.logger.info(f"  {i}. {feature}: {weight:.4f}")
                            component_analysis_data.append({
                                'Component': pc,
                                'Rank': i,
                                'Feature': feature,
                                'Weight': weight
                            })

                    # Create PCA component analysis report
                    component_analysis_df = pd.DataFrame(component_analysis_data)
                    component_analysis_df.to_csv(os.path.join(output_dir, 'pca_component_analysis.csv'), index=False)

                    # Visualizing the first two principal components
                    plt.figure(figsize=(10, 8))
                    try:
                        pc1 = components_df.iloc[0]
                        pc2 = components_df.iloc[1]

                        plt.scatter(pc1, pc2)

                        #
                        for i, feature in enumerate(components_df.columns[:30]):  #
                            plt.annotate(feature, (pc1[i], pc2[i]), fontsize=8)

                        plt.title('The weights of the original features on the first two principal components', fontsize=14)
                        plt.xlabel('PC1weight', fontsize=12)
                        plt.ylabel('PC2weight', fontsize=12)
                        plt.grid(True, alpha=0.3)
                        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, 'pca_components_analysis.png'), dpi=300,
                                    bbox_inches='tight')
                        plt.close()


                    except Exception as e:
                        self.logger.error(f"Error when visualizing PCA components: {e}")
            else:
                self.logger.warning("Unable to analyze feature importance: There is no feature importance data in the Random Forest model results")
        except Exception as e:
            self.logger.error(f"Error when analyzing IMU data significance: {str(e)}")

    def plot_enhanced_observation_behavior_histogram(self, df: pd.DataFrame = None, output_dir: str = 'results'):
        """
        Generates an enhanced histogram of observed behavior distribution, including counts and percentages

        Parameters:
            df (pd.DataFrame, optional): Data containing observed behavior labels. If not provided, labeled data is used
            output_dir (str): save

        Returns:
            pd.DataFrame: Observe behavioral statistics
        """
        if df is None:
            if self.labeled_data is None:
                raise ValueError("No data provided and no pre-labeled data")
            df = self.labeled_data

        # Check if the observed behavior column exists
        if 'observation' not in df.columns:
            self.logger.warning("Not found in data'observation'column")
            return None

        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Count the number and percentage of each observed behavior
        observation_counts = df['observation'].value_counts()
        total_count = len(df)
        observation_percentages = (observation_counts / total_count * 100).round(2)

        # Create a DataFrame containing counts and percentages
        stats_df = pd.DataFrame({
            'observation': observation_counts.index,
            'count': observation_counts.values,
            'percentage': observation_percentages.values
        })

        # Print Statistics
        self.logger.info("\nObservation behavior distribution:")
        self.logger.info("=" * 50)
        for i, row in stats_df.iterrows():
            # Convert floating point numbers to integers and then use integer format
            self.logger.info(f"{row['observation']:15}: {int(row['count']):4d} 段 ({row['percentage']:5.2f}%)")
        self.logger.info("=" * 50)
        self.logger.info(f"{'总计':15}: {total_count:4d} 段 (100.00%)")

        try:
            # Create subplots: one showing counts and one showing percentages
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Sub-figure 1: Display counts
            bars1 = ax1.bar(range(len(stats_df)), stats_df['count'].values,
                            color=plt.cm.Set3(np.linspace(0, 1, len(stats_df))))
            ax1.set_title('Observation behavior - number of segments', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Observing behavior types', fontsize=12)
            ax1.set_ylabel('number of segments', fontsize=12)
            ax1.set_xticks(range(len(stats_df)))
            ax1.set_xticklabels(stats_df['observation'].values, rotation=45, ha='right')

            # Add value labels to bar charts
            for i, (bar, count) in enumerate(zip(bars1, stats_df['count'].values)):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stats_df['count'].values) * 0.01,
                         f'{int(count)}', ha='center', va='bottom', fontweight='bold')

            ax1.grid(axis='y', alpha=0.3)

            # Sub-figure 2: Display percentage
            bars2 = ax2.bar(range(len(stats_df)), stats_df['percentage'].values,
                            color=plt.cm.Set3(np.linspace(0, 1, len(stats_df))))
            ax2.set_title('Observing behavior - percentage distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Observing behavior types', fontsize=12)
            ax2.set_ylabel('percentage (%)', fontsize=12)
            ax2.set_xticks(range(len(stats_df)))
            ax2.set_xticklabels(stats_df['observation'].values, rotation=45, ha='right')

            # Adding percentage labels to bar charts
            for i, (bar, percentage) in enumerate(zip(bars2, stats_df['percentage'].values)):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(stats_df['percentage'].values) * 0.01,
                         f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')

            ax2.grid(axis='y', alpha=0.3)
            ax2.set_ylim(0, max(stats_df['percentage'].values) * 1.1)

            # save
            plt.tight_layout()

            #
            combined_path = os.path.join(output_dir, 'observation_behavior_distribution_enhanced.png')
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"\nEnhanced histogram saved to: {combined_path}")

            return stats_df

        except Exception as e:
            self.logger.error(f"Error creating observed behavior histogram: {str(e)}")
            return stats_df


    def identify_position(self, lat: float, lon: float, position_boundaries: Dict = None) -> str:
        """
        Identify the location (P1, P2, or P3) based on latitude/longitude and assign the boundary point to the nearest region

        Parameters:
            lat (float): latitude
            lon (float): longitude
            position_boundaries (dict, optional): Location Boundary

        Returns:
            str: position identifier 'P1', 'P2', or 'P3'
        """
        # If location bounds are not provided, default values are used.
        if position_boundaries is None:
            return 'unknown'

        try:
            # First check if the point is in any region
            for pos, bounds in position_boundaries.items():
                if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                        bounds['lon_min'] <= lon <= bounds['lon_max']):
                    return pos

            # If not in any region, find the nearest region
            min_distance = float('inf')
            nearest_pos = 'unknown'

            for pos, bounds in position_boundaries.items():
                # Calculate the minimum distance to the region
                lat_dist = 0
                if lat < bounds['lat_min']:
                    lat_dist = bounds['lat_min'] - lat
                elif lat > bounds['lat_max']:
                    lat_dist = lat - bounds['lat_max']

                lon_dist = 0
                if lon < bounds['lon_min']:
                    lon_dist = bounds['lon_min'] - lon
                elif lon > bounds['lon_max']:
                    lon_dist = lon - bounds['lon_max']

                # Using Euclidean distance
                distance = (lat_dist ** 2 + lon_dist ** 2) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    nearest_pos = pos

            return nearest_pos
        except Exception as e:
            self.logger.error(f"Error identifying location: {str(e)}")
            return 'unknown'

    def identify_lane(self, lat: float, lon: float, bearing: float) -> str:
        """
        Identify lanes (L1 or L2) based on latitude/longitude and azimuth

        Parameters:
            lat (float): latitude
            lon (float): longitude
            bearing (float): Azimuth

        Returns:
            str: Lane Identifier 'L1' or 'L2'
        """
        try:
            # Classification based on azimuth
            if 0 <= bearing < 180:
                return 'L1'
            else:
                return 'L2'
        except Exception as e:
            self.logger.error(f"Error identifying lane: {str(e)}")
            return 'unknown'

    def track_position_lane_changes(self, df: pd.DataFrame = None,
                                    position_boundaries: Dict = None) -> pd.DataFrame:
        """
        Tracking position and lane changes

        Parameters:
            df (pd.DataFrame, optional): DataFrame with latitude, longitude, azimuth
            position_boundaries (dict, optional): position boundaries

        Returns:
            pd.DataFrame: DataFrame with added location, lane, and change identifiers
        """
        self.logger.info("Identify position and lane changes...")

        if df is None:
            if self.helmet_data is None:
                raise ValueError("No data provided and no data preloaded")
            df = self.helmet_data

        # copy dataframe
        data = df.copy()

        # Initialize position and lane columns
        data['position'] = 'unknown'
        data['lane'] = 'unknown'
        data['position_change'] = 'no_change'
        data['lane_change'] = 'no_change'

        try:
            # Use vectorized methods to handle positioning and create helper functions
            def get_position(row: pd.Series) -> str:
                if 'latitude' in row and 'longitude' in row:
                    return self.identify_position(row['latitude'], row['longitude'], position_boundaries)
                return 'unknown'

            def get_lane(row: pd.Series) -> str:
                if 'latitude' in row and 'longitude' in row:
                    if 'bearing' in row:
                        return self.identify_lane(row['latitude'], row['longitude'], row['bearing'])
                    elif 'speed' in row:
                        # If there is no azimuth but there is velocity, try to estimate it from the velocity direction
                        return 'L1' if row['speed'] >= 0 else 'L2'
                return 'unknown'

            # Vectorized position and lane recognition (more efficient for large datasets)
            if all(col in data.columns for col in ['latitude', 'longitude']):
                # Using apply is more efficient than looping
                self.logger.info("Application location identification...")
                data['position'] = data.apply(get_position, axis=1)

                self.logger.info("Applying lane recognition...")
                data['lane'] = data.apply(get_lane, axis=1)

                # Tracking location changes
                position_changed = data['position'] != data['position'].shift()
                valid_positions = (data['position'] != 'unknown') & (data['position'].shift() != 'unknown')
                position_change_mask = position_changed & valid_positions

                # Using vectorized operations to create position change labels
                data.loc[position_change_mask, 'position_change'] = (
                    data['position'].shift().str.cat(data['position'], sep='→')
                )[position_change_mask]

                # Tracking lane changes
                lane_changed = data['lane'] != data['lane'].shift()
                valid_lanes = (data['lane'] != 'unknown') & (data['lane'].shift() != 'unknown')
                lane_change_mask = lane_changed & valid_lanes

                # Creating lane change labels using vectorized operations
                data.loc[lane_change_mask, 'lane_change'] = (
                    data['lane'].shift().str.cat(data['lane'], sep='→')
                )[lane_change_mask]

            return data

        except Exception as e:
            self.logger.error(f"Errors in tracking position and lane changes: {str(e)}")
            return data

    def add_helmet_actions_to_excel(self, input_file_path: str,
                                    bike_file_path: str = None,
                                    output_file_path: str = None) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
        """
        Added columns for helmet movement, position change, and lane change to the original Excel file,
        Supports using bicycle data to improve recognition accuracy

        Parameters:
            input_file_path (str): Original helmet data Excel file path
            bike_file_path (str, optional): Original bike data Excel file path
            output_file_path (str, optional)

        Returns:
            tuple: (The saved Excel file path, DataFrame containing the action)
        """
        self.logger.info(f"Processing files: {input_file_path}")

        # If no output path is specified, a default path is created
        if output_file_path is None:
            file_name = os.path.splitext(input_file_path)[0]
            output_file_path = f"{file_name}_with_actions.xlsx"

        try:
            # Loading helmet data
            df = self.load_helmet_data(input_file_path)
            if df is None:
                self.logger.error("Unable to load data, unable to add helmet action column")
                return None, None

            # Load bicycle data (if any)
            bike_df = None
            if bike_file_path:
                bike_df = self.load_helmet_data(bike_file_path)
                if bike_df is None:
                    self.logger.warning("Unable to load bike data, only helmet data will be used")
                else:
                    self.bike_data = bike_df

            # Create Features
            df_with_features = self.create_helmet_features(df)

            # If there is bicycle data, then integrate
            if bike_df is not None:
                df_with_features = self.integrate_with_bike_data(df_with_features, bike_df)

            # Marking Observation Behavior
            labeled_data = self.label_observation_behaviors(df_with_features)
            self.labeled_data = labeled_data

            # Generate a histogram of observed behavior segments
            self.plot_enhanced_observation_behavior_histogram(labeled_data, output_dir='results')

            # Extracting observed behavior columns from labeled data
            observations = labeled_data['observation']

            # Add to original data
            df['helmet_action'] = observations.values

            # Calculate counts and percentages for each observed behavior
            action_counts = df['helmet_action'].value_counts()
            total = len(df)

            self.logger.info("\nHelmet motion distribution:")
            for action, count in action_counts.items():
                percentage = count / total * 100
                self.logger.info(f"{action}: {count} column ({percentage:.2f}%)")

            # Analyze GPS data range and calculate location boundaries
            if all(col in df.columns for col in ['latitude', 'longitude']):
                self.logger.info("\nAnalyze GPS data and mark locations and lanes...")
                lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
                lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

                self.logger.info(f"GPS data range:")
                self.logger.info(f"Latitude range: {lat_min:.6f} - {lat_max:.6f}")
                self.logger.info(f"Longitude range: {lon_min:.6f} - {lon_max:.6f}")

                # Calculate location boundaries with overlap to improve recognition rate
                lat_range = lat_max - lat_min
                lon_range = lon_max - lon_min
                expansion_factor = self.config['expansion_factor']  # Expansion Factor

                position_boundaries = {
                    'P1': {
                        'lat_min': lat_min,
                        'lat_max': lat_min + lat_range / 3 * (1 + expansion_factor),
                        'lon_min': lon_min,
                        'lon_max': lon_min + lon_range / 3 * (1 + expansion_factor)
                    },
                    'P2': {
                        'lat_min': lat_min + lat_range / 3 * (1 - expansion_factor),
                        'lat_max': lat_min + 2 * lat_range / 3 * (1 + expansion_factor / 2),
                        'lon_min': lon_min + lon_range / 3 * (1 - expansion_factor),
                        'lon_max': lon_min + 2 * lon_range / 3 * (1 + expansion_factor / 2)
                    },
                    'P3': {
                        'lat_min': lat_min + 2 * lat_range / 3 * (1 - expansion_factor / 2),
                        'lat_max': lat_max,
                        'lon_min': lon_min + 2 * lon_range / 3 * (1 - expansion_factor / 2),
                        'lon_max': lon_max
                    }
                }

                # Marking locations and lanes
                df_with_position_lane = self.track_position_lane_changes(df, position_boundaries)

                # Merge position and lane information into the original data
                for column in ['position', 'lane', 'position_change', 'lane_change']:
                    if column in df_with_position_lane.columns:
                        df[column] = df_with_position_lane[column].values

                # Print position and lane change statistics
                self.logger.info("\nposition distribution:")
                position_counts = df['position'].value_counts()
                for pos, count in position_counts.items():
                    self.logger.info(f"{pos}: {count} column ({count / total * 100:.2f}%)")

                self.logger.info("\nposition distribution:")
                position_changes = df[df['position_change'] != 'no_change']['position_change'].value_counts()
                for change, count in position_changes.items():
                    self.logger.info(f"{change}: {count} column")

                self.logger.info("\nLane distribution:")
                lane_counts = df['lane'].value_counts()
                for lane, count in lane_counts.items():
                    self.logger.info(f"{lane}: {count} column ({count / total * 100:.2f}%)")

                self.logger.info("\nlane distribution:")
                lane_changes = df[df['lane_change'] != 'no_change']['lane_change'].value_counts()
                for change, count in lane_changes.items():
                    self.logger.info(f"{change}: {count} column")
            else:
                self.logger.warning("The data lacks latitude/longitude information and cannot identify the location and lane")

            #
            df.to_excel(output_file_path, index=False)
            self.logger.info(f"Helmet movement, position and lane data saved to: {output_file_path}")
            return output_file_path, df

        except Exception as e:
            self.logger.error(f"Error saving Excel file: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None, None

    def visualize_gps_with_actions_and_positions(self, df: pd.DataFrame, output_path: str) -> Optional[str]:
        """
        Plot GPS tracks with helmet motion, position and lane changes to generate interactive maps

        Parameters:
        df (pd.DataFrame): Data with GPS and helmet motion
        output_path (str): Output HTML file path

        Returns:
        Optional[str]: Saved map path, or None if failed
        """
        # Check if the required column exists
        if not all(col in df.columns for col in ['latitude', 'longitude']):
            self.logger.error("Data is missing required columns(latitude, longitude)，Unable to draw map")
            return None

        self.logger.info("Create GPS track visualizations with helmet motion, position, and lane changes...")

        try:
            # Create a map centered on the average location of the data
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

            # Define colors for different helmet actions
            action_colors = {
                'forward': 'blue',
                'left_side': 'green',
                'right_side': 'purple',
                'left_mirror': 'orange',
                'right_mirror': 'yellow',
                'far': 'red',
                'near': 'black',
                'lifesaver_left': 'pink',
                'lifesaver_right': 'brown',
                'mid_level': 'gray'
            }

            # Define colors for different positions
            position_colors = {
                'P1': 'red',
                'P2': 'green',
                'P3': 'blue',
                'unknown': 'gray'
            }

            # Plotting GPS tracks
            coords = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
            folium.PolyLine(coords, color='blue', weight=3, opacity=0.7).add_to(m)

            # Add start and end markers
            folium.Marker(
                coords[0],
                popup='Starting point',
                icon=folium.Icon(color='green', icon='play', prefix='fa')
            ).add_to(m)

            folium.Marker(
                coords[-1],
                popup='Finish line',
                icon=folium.Icon(color='red', icon='stop', prefix='fa')
            ).add_to(m)

            # By location segment tag
            if 'helmet_action' in df.columns:
                #
                action_changes = df['helmet_action'].ne(df['helmet_action'].shift()).cumsum()
                action_segments = df.groupby(action_changes)

                for _, segment in action_segments:
                    action = segment['helmet_action'].iloc[0]
                    color = action_colors.get(action, 'gray')

                    # Create a list of coordinates for the segment
                    segment_coords = [[row['latitude'], row['longitude']] for _, row in segment.iterrows()]

                    # Draw the trajectory
                    folium.PolyLine(
                        segment_coords,
                        color=color,
                        weight=5,
                        opacity=0.8,
                        popup=f"Action: {action}, Duration: {len(segment)}frame"
                    ).add_to(m)

            # Display location partitions
            if 'position' in df.columns:
                # Use grouping method to label location partitions
                position_changes = df['position'].ne(df['position'].shift()).cumsum()
                position_segments = df.groupby(position_changes)

                for _, segment in position_segments:
                    position = segment['position'].iloc[0]
                    if position != 'unknown':
                        # Create another track for this location partition
                        position_coords = [[row['latitude'], row['longitude']] for _, row in segment.iterrows()]

                        # Add location tags to the map
                        center_point = position_coords[len(position_coords) // 2]
                        folium.Marker(
                            center_point,
                            popup=f"position: {position}",
                            icon=folium.Icon(color=position_colors.get(position, 'gray'), icon='map-marker',
                                             prefix='fa')
                        ).add_to(m)

            # Add a legend
            legend_html = """
            <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; font-size:14px; background-color: white; padding: 10px; border: 1px solid grey; border-radius: 5px;">
            <p><b>Helmet movement illustration:</b></p>
            """

            for action, color in action_colors.items():
                legend_html += f"""
                <p><i class="fa fa-circle" style="color:{color}"></i> {action}</p>
                """

            if 'position' in df.columns:
                legend_html += "<p><b>Location legend:</b></p>"
                for position, color in position_colors.items():
                    if position != 'unknown':
                        legend_html += f"""
                        <p><i class="fa fa-map-marker" style="color:{color}"></i> {position}</p>
                        """

            legend_html += "</div>"

            m.get_root().html.add_child(folium.Element(legend_html))

            # save
            m.save(output_path)
            self.logger.info(f"Map saved to: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating map visualization: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def predict_observation(self, model, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Use the trained model to predict helmet observation behavior

       Parameters:
       model: training model pipeline
       new_data (pd.DataFrame): new helmet IMU data

       Returns:
       pd.DataFrame: data with predicted observation behavior
        """
        # Preprocessing new data
        self.logger.info("Using models to predict observed behavior...")
        processed_data = self.create_helmet_features(new_data)

        # 检查NaN值
        if processed_data.isna().any().any():
            self.logger.warning("The preprocessed data contains NaN values, which are filled with column means....")
            for col in processed_data.columns:
                if processed_data[col].isna().any():
                    col_mean = processed_data[col].mean()
                    if np.isnan(col_mean):  # If the entire column is NaN
                        processed_data[col] = 0
                    else:
                        processed_data[col] = processed_data[col].fillna(col_mean)

        try:
            # Extract all features used by the model
            pipeline_steps = {step[0]: step[1] for step in model.steps}
            has_pca = 'pca' in pipeline_steps

            # Choose appropriate feature columns
            feature_cols = []

            # Helmet IMU Features
            imu_prefixes = ['a', 'w', 'total_acc', 'total_angular_vel']
            imu_suffixes = ['', '_diff', '_mean', '_std', '_max', '_min', '_range']

            for prefix in imu_prefixes:
                for axis in ['x', 'y', 'z']:
                    for suffix in imu_suffixes:
                        col = f'{prefix}{axis}{suffix}'
                        if col in processed_data.columns:
                            feature_cols.append(col)

            # Adding other features
            for col in processed_data.columns:
                if (col.startswith('q') and '_' in col) or \
                        col.endswith('_duration') or \
                        '_dom_freq' in col or \
                        '_spectral_' in col:
                    feature_cols.append(col)

            # Check if the feature list is empty
            if not feature_cols:
                self.logger.warning("Warning: No suitable features found for prediction")
                return processed_data

            # Only keep the columns that are actually present in the final feature list
            available_features = [col for col in feature_cols if col in processed_data.columns]
            self.logger.info(f"use {len(available_features)} features to predict")

            # to prediction
            X_new = processed_data[available_features]
            predictions = model.predict(X_new)

            # Add predictions to the dataframe
            processed_data['predicted_observation'] = predictions

            # Print forecast statistics
            pred_counts = processed_data['predicted_observation'].value_counts()
            total = len(processed_data)

            self.logger.info("\nPredicting observed behavior distribution:")
            for behavior, count in pred_counts.items():
                self.logger.info(f"{behavior}: {count} ({count / total * 100:.2f}%)")

            return processed_data

        except Exception as e:
            self.logger.error(f"Errors in the prediction process: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return processed_data

    def save_models(self, output_dir: str = 'models'):
        """
        Save all trained models

        Parameters:
        output_dir (str): The directory where the model is saved
        """
        self.logger.info(f"Save the model to a directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Save Observation Behavior Model
            if hasattr(self, 'observation_models'):
                for name, model in self.observation_models.items():
                    model_filename = os.path.join(output_dir, f"observation_{name.replace(' ', '_')}_model.pkl")
                    joblib.dump(model, model_filename)
                    self.logger.info(f"The observed behavior recognition model has been saved as {model_filename}")

            # Save the bicycle motion recognition model
            if hasattr(self, 'bike_motion_models'):
                for name, model in self.bike_motion_models.items():
                    model_filename = os.path.join(output_dir, f"bike_motion_{name.replace(' ', '_')}_model.pkl")
                    joblib.dump(model, model_filename)
                    self.logger.info(f"The bicycle motion recognition model has been saved as {model_filename}")

            # Save the configuration file
            config_filename = os.path.join(output_dir, "config.json")
            f: TextIO
            with open(config_filename, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f"The configuration file has been saved as {config_filename}")

            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    def load_models(self, model_dir: str = 'models'):
        """
        Load pre-trained model

        Parameters:
        model_dir (str): Directory containing the model

        Returns:
        bool: True if loaded successfully, False otherwise
        """
        self.logger.info(f"Loading models from a directory: {model_dir}")

        try:
            # Loading a Configuration File
            config_filename = os.path.join(model_dir, "config.json")
            if os.path.exists(config_filename):
                with open(config_filename, 'r') as f:
                    self.config = json.load(f)
                self.logger.info(f"Configuration file loaded: {config_filename}")

            # Observational Behavior Model
            observation_models = {}
            for model_name in ["Random_Forest", "SVM", "Neural_Network"]:
                model_filename = os.path.join(model_dir, f"observation_{model_name}_model.pkl")
                if os.path.exists(model_filename):
                    model = joblib.load(model_filename)
                    observation_models[model_name.replace('_', ' ')] = model
                    self.logger.info(f"Observation behavior model loaded: {model_filename}")

            if observation_models:
                self.observation_models = observation_models

            # Bicycle sports model
            bike_motion_models = {}
            for model_name in ["Random_Forest", "SVM", "Neural_Network"]:
                model_filename = os.path.join(model_dir, f"bike_motion_{model_name}_model.pkl")
                if os.path.exists(model_filename):
                    model = joblib.load(model_filename)
                    bike_motion_models[model_name.replace('_', ' ')] = model
                    self.logger.info(f"Cycling model loaded: {model_filename}")

            if bike_motion_models:
                self.bike_motion_models = bike_motion_models

            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False

    def run_complete_analysis(self, helmet_file_path: str,
                              bike_file_path: str = None,
                              output_dir: str = 'results',
                              use_gps: bool = True) -> bool:
        """
        Complete analysis process: load data, train model, analyze results

        Parameters:
        helmet_file_path (str): Helmet IMU data file path
        bike_file_path (str, optional): Bike IMU data file path, optional
        output_dir (str): Output directory path
        use_gps (bool): Whether to perform GPS related analysis, the default is True

        Returns:
         bool: True if the analysis is successful, otherwise False
        """
        self.logger.info("===== Helmet-based Observation Behavior Analysis System =====")

        # Create Output Directory
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Output Directory: {output_dir}")

        try:
            # 1. load helmet data
            helmet_df = self.load_helmet_data(helmet_file_path)
            if helmet_df is None:
                self.logger.error("Unable to load helmet data, program exited")
                return False

            # 2. Load bicycle data (if any)
            bike_df = None
            if bike_file_path:
                bike_df = self.load_helmet_data(bike_file_path)
                if bike_df is None:
                    self.logger.warning("Unable to load bike data, only helmet data will be used")
                else:
                    self.bike_data = bike_df

            # 3. Creating the Helmet Feature
            helmet_with_features = self.create_helmet_features(helmet_df)

            # Check if there are NaN values after feature creation
            nan_count = helmet_with_features.isna().sum().sum()
            if nan_count > 0:
                self.logger.warning(f"Warning: After feature creation there is {nan_count} NaN values")
                self.logger.info("Fill NaN values with column mean...")

                # fill NaN values
                for col in helmet_with_features.columns:
                    if helmet_with_features[col].isna().any():
                        col_mean = helmet_with_features[col].mean()
                        if np.isnan(col_mean):  # If the entire column is NaN
                            helmet_with_features[col] = 0
                        else:
                            helmet_with_features[col] = helmet_with_features[col].fillna(col_mean)

            # 4. Integrate helmet and bike data (where available)
            if bike_df is not None:
                integrated_data = self.integrate_with_bike_data(helmet_with_features, bike_df)
            else:
                integrated_data = helmet_with_features

            # 5. Tagging Helmet Observation Behavior
            labeled_data = self.label_observation_behaviors(integrated_data)
            self.labeled_data = labeled_data

            # Create an Observational Behavior Chart - Keep only the enhanced version to avoid duplicate output
            self.plot_enhanced_observation_behavior_histogram(labeled_data, output_dir=output_dir)

            # 6. A) Training a helmet observation behavior recognition model - without using bicycle data
            self.logger.info("\nTraining a helmet observation behavior recognition model...")
            observation_models, observation_results = self.train_observation_models(
                labeled_data, use_bike_data=False, output_dir=output_dir
            )

            self.observation_models = observation_models
            self.observation_results = observation_results

            # 6. B) Train a helmet observation behavior recognition model - using bicycle data if available
            if bike_df is not None:
                self.logger.info("\nUsing bicycle data to train a helmet observation behavior recognition model...")
                observation_models_with_bike, observation_results_with_bike = self.train_observation_models(
                    labeled_data, use_bike_data=True, output_dir=output_dir
                )

                # Comparing the accuracy of data with and without bicycles
                self.logger.info("\nComparison of helmet observation behavior recognition accuracy (with/without bicycle data):")
                for model_name in observation_models.keys():
                    acc_without = observation_results[model_name]['accuracy']
                    acc_with = observation_results_with_bike[model_name]['accuracy']
                    diff = acc_with - acc_without
                    self.logger.info(
                        f"{model_name}: Without bike data {acc_without:.4f}, with bike data {acc_with:.4f}, difference {diff:.4f}"
                    )

                # If the accuracy using bicycle data is higher, update the model
                if any(observation_results_with_bike[m]['accuracy'] > observation_results[m]['accuracy']
                       for m in observation_models.keys()):
                    self.logger.info("The model using bicycle data is more accurate and updates the best model")
                    self.observation_models = observation_models_with_bike
                    self.observation_results = observation_results_with_bike

            # 7. Analyze position and lane changes - only if use_gps is True and geolocation data exists
            helmet_df_with_positions = None
            if use_gps and all(col in helmet_df.columns for col in ['latitude', 'longitude']):
                self.logger.info("\nPerform GPS correlation analysis...")
                lat_min, lat_max = helmet_df['latitude'].min(), helmet_df['latitude'].max()
                lon_min, lon_max = helmet_df['longitude'].min(), helmet_df['longitude'].max()

                self.logger.info(f"GPS data range:")
                self.logger.info(f"Latitude range: {lat_min:.6f} - {lat_max:.6f}")
                self.logger.info(f"Longitude range: {lon_min:.6f} - {lon_max:.6f}")

                # Calculate location boundaries with overlap to improve recognition rate
                lat_range = lat_max - lat_min
                lon_range = lon_max - lon_min
                expansion_factor = self.config['expansion_factor']

                position_boundaries = {
                    'P1': {
                        'lat_min': lat_min,
                        'lat_max': lat_min + lat_range / 3 * (1 + expansion_factor),
                        'lon_min': lon_min,
                        'lon_max': lon_min + lon_range / 3 * (1 + expansion_factor)
                    },
                    'P2': {
                        'lat_min': lat_min + lat_range / 3 * (1 - expansion_factor),
                        'lat_max': lat_min + 2 * lat_range / 3 * (1 + expansion_factor / 2),
                        'lon_min': lon_min + lon_range / 3 * (1 - expansion_factor),
                        'lon_max': lon_min + 2 * lon_range / 3 * (1 + expansion_factor / 2)
                    },
                    'P3': {
                        'lat_min': lat_min + 2 * lat_range / 3 * (1 - expansion_factor / 2),
                        'lat_max': lat_max,
                        'lon_min': lon_min + 2 * lon_range / 3 * (1 - expansion_factor / 2),
                        'lon_max': lon_max
                    }
                }

                helmet_df_with_positions = self.track_position_lane_changes(helmet_df, position_boundaries)
            elif use_gps:
                self.logger.warning(
                    "\nWarning: Data lacks geographic location information (latitude, longitude), skipping GPS related analysis"
                )
            else:
                self.logger.info("\nGPS analysis disabled, skipping position and lane change analysis")

            # 8. Recognizing Cycling Motion from Helmet Data
            self.logger.info("\nTraining a bicycle motion recognition model...")
            bike_motion_models, bike_motion_results = self.train_bike_motion_from_helmet(
                labeled_data, output_dir=output_dir
            )

            self.bike_motion_models = bike_motion_models
            self.bike_motion_results = bike_motion_results

            # 9. Analyze the importance of helmet IMU data - save the results to the output directory
            self.analyze_helmet_imu_importance(observation_results, output_dir=output_dir)

            # 10. Save the model to the output directory
            model_output_dir = os.path.join(output_dir, "models")
            self.save_models(model_output_dir)

            # 11. Generate fully labeled data - save to output directory
            complete_output_path = os.path.join(output_dir, "helmet_data_complete.xlsx")
            if use_gps:
                helmet_file_output, helmet_df_with_actions = self.add_helmet_actions_to_excel(
                    helmet_file_path, bike_file_path, complete_output_path
                )

                # Create a visualization map and save it to the output directory
                if helmet_df_with_actions is not None:
                    self.visualize_gps_with_actions_and_positions(
                        helmet_df_with_actions,  # Pass in a DataFrame instead of a file path
                        os.path.join(output_dir, "gps_visualization.html")
                    )
            else:
                # Only add observed behavior labels, excluding location and lane information
                df = helmet_df.copy()
                df['helmet_action'] = labeled_data['observation'].values
                df.to_excel(complete_output_path, index=False)
                self.logger.info(f"Data with helmet-observed behavior labels has been saved to: {complete_output_path}")

            # 12. summary the results
            self.logger.info("\n===== summary =====")

            # Best observed behavior recognition model
            best_obs_model = max(observation_results.items(), key=lambda x: x[1]['accuracy'])[0]
            best_obs_acc = observation_results[best_obs_model]['accuracy']
            self.logger.info(f"The best helmet-observed behavior recognition model: {best_obs_model}, accuracy: {best_obs_acc:.4f}")

            # Best bike motion recognition model
            best_bike_model = max(bike_motion_results.items(), key=lambda x: x[1]['accuracy'])[0]
            best_bike_acc = bike_motion_results[best_bike_model]['accuracy']
            self.logger.info(f"Best bike motion recognition model: {best_bike_model}, accuracy: {best_bike_acc:.4f}")

            # Conclusions about position and lane (only if GPS analysis was performed)
            if use_gps and helmet_df_with_positions is not None:
                self.logger.info("\nConclusions about Position and Lane Recognition:")
                self.logger.info(
                    "1. Location recognition divides the route into three areas (P1, P2, P3) to monitor location changes"
                )
                self.logger.info(
                    "2. Lane recognition uses azimuth to identify lanes L1 and L2, and realizes lane change monitoring"
                )
                self.logger.info(
                    "3. Visual map shows the position and lane changes intuitively"
                )

            # Prints a summary of all files stored in the output directory
            self.logger.info("\nAll files saved in the output directory:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path) / 1024  # (KB)
                    self.logger.info(f"- {file} ({file_size:.1f} KB)")

            return True

        except Exception as e:
            self.logger.error(f"Error while training model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


# If you run this script directly
if __name__ == "__main__":

    helmet_file_path = r"helmet_data.xlsx"
    default_output_dir = r"E:\python_job\results"
    bike_file_path = r"bike_data.xlsx"


    # Ask whether to use GPS analysis
    use_gps = input("Whether to perform GPS related analysis？(y/n): ").strip().lower() == 'y'
    if use_gps:
        print("GPS related analysis will be conducted")
    else:
        print("No GPS related analysis")

    # Configure output directory
    output_dir = input(f"Output directory (press Enter to use the default value: {default_output_dir}): ").strip()
    if not output_dir:
        output_dir = default_output_dir

    # Configuring Logging
    logger = setup_logger()
    logger.info("===== Helmet-based Observation Behavior Analysis System =====")
    logger.info(f"Helmet data: {helmet_file_path}")
    logger.info(f"Bike data: {bike_file_path}")
    logger.info(f"GPS analysis: {'Enable' if use_gps else 'Disable'}")
    logger.info(f"Output Directory: {output_dir}")

    analyzer = HelmetAnalyzer()

    # Run the analysis
    success = analyzer.run_complete_analysis(
        helmet_file_path,
        bike_file_path,
        output_dir=output_dir,
        use_gps=use_gps
    )

    if success:
        logger.info("Analysis Complete！")
    else:
        logger.error("Error during analysis")

    # Prevent the window from closing immediately
    input("Press Enter to exit...")