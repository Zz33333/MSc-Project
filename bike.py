import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import time
import joblib
from matplotlib.font_manager import FontProperties
import os

# Create results directory if it doesn't exist
results_dir = "result"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")

# Configure matplotlib for English display
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = True


# 1. Data loading and preprocessing
def load_data(file_path):
    """
    Load and preprocess bicycle motion data

    Parameters:
        file_path (str): Excel file path
    Returns:
        pd.DataFrame: Preprocessed data
    """
    print(f"Loading data: {file_path}")
    df = pd.read_excel(file_path)

    # Display basic information
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Check and handle missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Processing missing values...")
        # Use forward fill for time series
        df = df.fillna(method='ffill')
        # Use backward fill for any remaining NaN values
        df = df.fillna(method='bfill')

    return df


# 2. Feature engineering
def create_features(df, use_gps=True):
    """
    Create features from raw data

    Parameters:
        df (pd.DataFrame): Raw data
        use_gps (bool): Whether to use GPS data
    Returns:
        pd.DataFrame: Data with features
    """
    print("Creating features...")
    data = df.copy()

    # IMU data processing
    if all(col in data.columns for col in ['ax', 'ay', 'az']):
        # Calculate total acceleration
        data['total_acc'] = np.sqrt(data['ax'] ** 2 + data['ay'] ** 2 + data['az'] ** 2)

        # Calculate acceleration change rate
        data['acc_change'] = data['total_acc'].diff().fillna(0)

        # Calculate rolling statistics for acceleration
        window = 5  # Rolling window size
        data['acc_rolling_mean'] = data['total_acc'].rolling(window=window, center=True).mean().ffill().bfill()
        data['acc_rolling_std'] = data['total_acc'].rolling(window=window, center=True).std().ffill().bfill()

        # Calculate statistics for each acceleration axis
        for axis in ['ax', 'ay', 'az']:
            data[f'{axis}_rolling_mean'] = data[axis].rolling(window=window, center=True).mean().ffill().bfill()
            data[f'{axis}_rolling_std'] = data[axis].rolling(window=window, center=True).std().ffill().bfill()

    # Angular velocity data processing
    if all(col in data.columns for col in ['wx', 'wy', 'wz']):
        # Calculate total angular velocity
        data['total_angular_vel'] = np.sqrt(data['wx'] ** 2 + data['wy'] ** 2 + data['wz'] ** 2)

        # Statistics for each angular velocity axis
        for axis in ['wx', 'wy', 'wz']:
            data[f'{axis}_rolling_mean'] = data[axis].rolling(window=window, center=True).mean().ffill().bfill()
            data[f'{axis}_rolling_std'] = data[axis].rolling(window=window, center=True).std().ffill().bfill()

    # GPS data processing - only execute when use_gps is True
    if use_gps and 'bearing' in data.columns:
        # Calculate bearing angle change
        data['bearing_prev'] = data['bearing'].shift(1).fillna(data['bearing'].iloc[0])

        # Consider the cyclical nature of bearing (0-360 degrees)
        data['bearing_change'] = np.minimum(
            np.abs(data['bearing'] - data['bearing_prev']),
            360 - np.abs(data['bearing'] - data['bearing_prev'])
        )

        # Rolling statistics for bearing change
        data['bearing_change_rolling_mean'] = data['bearing_change'].rolling(window=window,
                                                                             center=True).mean().ffill().bfill()
        data['bearing_change_rolling_std'] = data['bearing_change'].rolling(window=window,
                                                                            center=True).std().ffill().bfill()

    # Calculate FFT features (optional)
    # Here we only extract FFT features for total acceleration as an example
    if 'total_acc' in data.columns:
        # Divide data into small segments for FFT
        segment_size = 20  # Samples per segment
        n_segments = len(data) // segment_size

        fft_features = []
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size

            # Perform FFT on this segment
            segment = data.iloc[start_idx:end_idx]

            if 'total_acc' in segment.columns:
                fft_result = np.abs(np.fft.fft(segment['total_acc'].values))

                # Only take the first few frequency components as features
                n_components = 3
                fft_features.append(fft_result[:n_components])

        # If FFT features were successfully generated
        if fft_features:
            fft_features = np.array(fft_features)

            # Repeat FFT features to match original data length
            repeated_features = np.repeat(fft_features, segment_size, axis=0)

            # Handle possible length mismatch (last few samples)
            if len(repeated_features) > len(data):
                repeated_features = repeated_features[:len(data)]
            elif len(repeated_features) < len(data):
                # Fill remaining rows with the last group of features
                padding = np.repeat(fft_features[-1:], len(data) - len(repeated_features), axis=0)
                repeated_features = np.vstack([repeated_features, padding])

            # Add FFT features to dataframe
            for i in range(n_components):
                data[f'fft_acc_{i}'] = repeated_features[:, i]

    return data


# Add a function to calculate dynamic turn threshold
def calculate_dynamic_turn_threshold(speed, base_threshold=25, min_threshold=5, max_threshold=45):
    """
    Dynamically calculate turn threshold based on speed

    Parameters:
        speed (float): Current speed (km/h or m/s)
        base_threshold (float): Base threshold
        min_threshold (float): Minimum threshold
        max_threshold (float): Maximum threshold
    Returns:
        float: Dynamically adjusted turn threshold
    """
    # Use maximum threshold if speed is zero or negative
    if speed <= 0:
        return max_threshold

    # Increase threshold at lower speeds, decrease at higher speeds
    # Use inverse proportion: threshold = base_threshold * (speed_factor)
    # where speed_factor is larger at lower speeds, smaller at higher speeds

    # Define speed interval (can be adjusted according to actual situation)
    reference_speed = 10.0  # Reference speed point

    # Calculate speed factor (higher speed, smaller factor, smaller threshold)
    speed_factor = reference_speed / max(speed, 1.0)  # Avoid division by zero

    # Calculate dynamic threshold
    dynamic_threshold = base_threshold * speed_factor

    # Limit to min and max threshold range
    return max(min(dynamic_threshold, max_threshold), min_threshold)


# 3. Label behaviors
def label_behaviors(df, acc_threshold=0.1, decel_threshold=-0.1, base_turn_threshold=25, use_gps=True):
    """
    Label bicycle riding behaviors using dynamically adjusted turn thresholds

    Parameters:
        df (pd.DataFrame): Data with features
        acc_threshold (float): Acceleration threshold
        decel_threshold (float): Deceleration threshold
        base_turn_threshold (float): Base turning threshold
        use_gps (bool): Whether to use GPS data

    Returns:
        pd.DataFrame: Data with behavior labels
    """
    print("Labeling behaviors...")
    data = df.copy()

    # Initialize behavior column
    data['behavior'] = 'straight'  # Default is straight driving

    # Label acceleration and deceleration
    if 'acc_change' in data.columns:
        data.loc[data['acc_change'] > acc_threshold, 'behavior'] = 'acceleration'
        data.loc[data['acc_change'] < decel_threshold, 'behavior'] = 'deceleration'

    # Use dynamic turn threshold to label turns
    if use_gps and 'bearing_change' in data.columns:
        # Get speed data - if no speed column, try to calculate or use default value
        if 'speed' not in data.columns:
            print("Warning: Speed data not found, using estimated value")
            # Can try to calculate speed from position change, or use default value
            data['speed'] = 10.0  # Default speed value, unit might be km/h or m/s

        # Calculate dynamic turn threshold for each row and determine if it's a turn
        for i in range(len(data)):
            # Get current speed
            current_speed = data.loc[i, 'speed']

            # Calculate dynamic turn threshold at current speed
            dynamic_threshold = calculate_dynamic_turn_threshold(
                current_speed,
                base_threshold=base_turn_threshold
            )

            # Determine if it's a turn (higher priority than acceleration and deceleration)
            if data.loc[i, 'bearing_change'] > dynamic_threshold:
                data.loc[i, 'behavior'] = 'turning'

                # Optional: Record used threshold for analysis
                if 'turn_threshold_used' not in data.columns:
                    data['turn_threshold_used'] = None
                data.loc[i, 'turn_threshold_used'] = dynamic_threshold

    # If not using GPS data, try to detect turns using IMU data
    elif 'total_angular_vel' in data.columns:
        # For IMU data, can also try to dynamically adjust threshold based on speed
        base_angular_vel_threshold = 60  # Base angular velocity threshold

        # If speed data available, use dynamic threshold
        if 'speed' in data.columns:
            for i in range(len(data)):
                current_speed = data.loc[i, 'speed']
                # IMU angular velocity threshold can also be dynamically adjusted
                # Use simple linear relationship: higher speed, lower threshold
                dynamic_angular_threshold = base_angular_vel_threshold * (
                        1.0 - min(current_speed / 30.0, 0.7)  # Reduce by at most 70%
                )

                if data.loc[i, 'total_angular_vel'] > dynamic_angular_threshold:
                    data.loc[i, 'behavior'] = 'turning'

                    # Optional: Record used threshold
                    if 'angular_threshold_used' not in data.columns:
                        data['angular_threshold_used'] = None
                    data.loc[i, 'angular_threshold_used'] = dynamic_angular_threshold

        # If no speed data, use fixed threshold
        else:
            data.loc[data['total_angular_vel'] > base_angular_vel_threshold, 'behavior'] = 'turning'

    # Convert to categorical variable
    data['behavior'] = pd.Categorical(data['behavior'],
                                      categories=['straight', 'acceleration', 'deceleration', 'turning'])

    # Statistics of behavior distribution
    behavior_counts = data['behavior'].value_counts()
    print("\nBehavior Distribution:")
    for behavior, count in behavior_counts.items():
        print(f"{behavior}: {count} ({count / len(data) * 100:.2f}%)")

    return data


# 4. Feature selection
def select_features(df, use_gps=True):
    """
    Select features for training

    Parameters:
        df (pd.DataFrame): Data with features and labels
        use_gps (bool): Whether to use GPS data

    Returns:
        tuple: (X, y) Feature matrix and target vector
    """
    print("Selecting features...")

    # Potential feature groups
    imu_features = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
    angle_features = ['AngleX', 'AngleY', 'AngleZ']
    gps_features = [] if not use_gps else ['speed', 'bearing']

    # Derived features
    acc_features = [
        'total_acc', 'acc_change', 'acc_rolling_mean', 'acc_rolling_std',
        'ax_rolling_mean', 'ax_rolling_std', 'ay_rolling_mean', 'ay_rolling_std',
        'az_rolling_mean', 'az_rolling_std'
    ]

    angular_features = [
        'total_angular_vel',
        'wx_rolling_mean', 'wx_rolling_std', 'wy_rolling_mean', 'wy_rolling_std',
        'wz_rolling_mean', 'wz_rolling_std'
    ]

    bearing_features = [] if not use_gps else [
        'bearing_change', 'bearing_change_rolling_mean', 'bearing_change_rolling_std'
    ]

    # FFT features
    fft_features = [col for col in df.columns if 'fft_' in col]

    # All possible features
    all_potential_features = (
            imu_features + angle_features + gps_features +
            acc_features + angular_features + bearing_features + fft_features
    )

    # Select features present in the dataframe
    selected_features = [f for f in all_potential_features if f in df.columns]

    print(f"Selected {len(selected_features)} features:")
    print(", ".join(selected_features))

    # Return feature matrix and target vector
    X = df[selected_features]
    y = df['behavior']

    return X, y


# 5. Train and evaluate models
def train_and_evaluate_models(X, y):
    """
    Train and evaluate three different machine learning models

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector

    Returns:
        dict: Dictionary containing trained models
    """
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Define three different models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        ),

        "Support Vector Machine": SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        ),

        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=300,
            random_state=42
        )
    }

    # Create pipelines with preprocessing
    pipelines = {}
    trained_models = {}
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} model...")

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        # Record start time
        start_time = time.time()

        # Train model
        pipeline.fit(X_train, y_train)

        # Record training time
        train_time = time.time() - start_time

        # Record start time
        start_time = time.time()

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)

        # Record prediction time
        predict_time = time.time() - start_time

        # Store results
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'train_time': train_time,
            'predict_time': predict_time,
            'predict_time_per_sample': predict_time / len(X_test)
        }

        print(f"{name} model accuracy: {accuracy:.4f}")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Prediction time: {predict_time:.4f} seconds ({predict_time / len(X_test) * 1000:.4f} ms per sample)")

        pipelines[name] = pipeline
        trained_models[name] = pipeline.named_steps['classifier']

    # Plot comparison charts
    plot_model_comparisons(results)

    # Analyze importance of IMU data
    analyze_imu_importance(X, y, trained_models)

    return pipelines, results


# 6. Plot model comparisons
def plot_model_comparisons(results):
    """
    Plot model performance comparison charts

    Parameters:
        results (dict): Dictionary containing model evaluation results
    """
    # Accuracy comparison
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]

    bar_plot = plt.bar(model_names, accuracies, color=['blue', 'green', 'red'])

    # Display accuracy values on each bar
    for bar, acc in zip(bar_plot, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.4f}', ha='center', fontsize=11)

    plt.title('Accuracy Comparison of Three Models', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_accuracy_comparison.png'), dpi=300, bbox_inches='tight')

    # Time performance comparison
    plt.figure(figsize=(12, 6))
    train_times = [results[name]['train_time'] for name in model_names]
    predict_times = [results[name]['predict_time_per_sample'] * 1000 for name in model_names]  # Convert to milliseconds

    x = np.arange(len(model_names))
    width = 0.35

    plt.bar(x - width / 2, train_times, width, label='Training Time (seconds)', color='blue')
    plt.bar(x + width / 2, predict_times, width, label='Prediction Time per Sample (milliseconds)', color='orange')

    plt.title('Model Time Performance Comparison', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_time_comparison.png'), dpi=300, bbox_inches='tight')

    # Plot confusion matrix for each model
    for name, result in results.items():
        plt.figure(figsize=(8, 6))
        cm = result['confusion_matrix']

        class_names = ['Straight', 'Acceleration', 'Deceleration', 'Turning']

        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)

        plt.title(f'Confusion Matrix of {name} Model', fontsize=14)
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')


# 7. Analyze importance of IMU data
def analyze_imu_importance(X, y, models):
    """
    Analyze the importance of IMU data in different models

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        models (dict): Dictionary of trained models
    """
    print("\nAnalyzing the importance of IMU data...")

    # Feature importance only works for Random Forest
    if "Random Forest" in models:
        rf_model = models["Random Forest"]

        # Get feature importance
        feature_importances = rf_model.feature_importances_

        # Create feature importance dataframe
        importances_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)

        # Mark IMU features
        def is_imu_feature(feature_name):
            imu_prefixes = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
            return any(feature_name.startswith(prefix) for prefix in imu_prefixes) or \
                'total_acc' in feature_name or 'total_angular' in feature_name

        importances_df['Feature_Type'] = importances_df['Feature'].apply(
            lambda x: 'IMU' if is_imu_feature(x) else 'Other'
        )

        # Print overall importance of IMU features
        imu_importance = importances_df[importances_df['Feature_Type'] == 'IMU']['Importance'].sum()
        print(f"Overall importance of IMU features: {imu_importance:.4f} ({imu_importance * 100:.2f}%)")

        # Plot Top 15 feature importances
        plt.figure(figsize=(12, 8))
        top_features = importances_df.head(15)

        # Fix warning: Add hue and set legend=False to solve FutureWarning
        ax = sns.barplot(x='Importance', y='Feature', hue='Feature_Type', data=top_features,
                         palette={'IMU': 'blue', 'Other': 'gray'})

        plt.title('Top 15 Important Features (Random Forest)', fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.legend(title='Feature Type')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')

        # Return breakdown of IMU feature importances
        imu_features_importances = importances_df[importances_df['Feature_Type'] == 'IMU']
        print("\nIMU Feature Importance Ranking:")
        for i, (_, row) in enumerate(imu_features_importances.iterrows(), 1):
            print(f"{i}. {row['Feature']}: {row['Importance']:.4f}")

        # Analyze importance of IMU data in different axes
        axis_importance = {}
        for axis in ['x', 'y', 'z']:
            # Acceleration
            acc_cols = [col for col in importances_df['Feature'] if f'a{axis}' in col]
            acc_importance = importances_df[importances_df['Feature'].isin(acc_cols)]['Importance'].sum()
            axis_importance[f'Acceleration-{axis}'] = acc_importance

            # Angular velocity
            gyro_cols = [col for col in importances_df['Feature'] if f'w{axis}' in col]
            gyro_importance = importances_df[importances_df['Feature'].isin(gyro_cols)]['Importance'].sum()
            axis_importance[f'Angular Velocity-{axis}'] = gyro_importance

        # Plot axis importance comparison chart
        plt.figure(figsize=(10, 6))
        axes = list(axis_importance.keys())
        values = list(axis_importance.values())

        colors = ['blue', 'blue', 'blue', 'green', 'green', 'green']

        # Fix warning: Use hue parameter
        ax_types = ['Acceleration'] * 3 + ['Angular Velocity'] * 3
        ax_df = pd.DataFrame({
            'IMU Sensor Axis': axes,
            'Cumulative Importance': values,
            'Type': ax_types
        })

        # Use hue parameter
        sns.barplot(x='IMU Sensor Axis', y='Cumulative Importance', hue='Type', data=ax_df,
                    palette={'Acceleration': 'blue', 'Angular Velocity': 'green'})

        plt.title('IMU Data Axis Importance Comparison', fontsize=14)
        plt.xlabel('IMU Sensor Axis', fontsize=12)
        plt.ylabel('Cumulative Importance', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'imu_axis_importance.png'), dpi=300, bbox_inches='tight')

    else:
        print("Cannot analyze feature importance: Random Forest model is not available")


# Modified position identification function - updated to use nearest region assignment algorithm
def identify_position(lat, lon, position_boundaries=None):
    """
    Identify position (P1, P2 or P3) based on latitude and longitude,
    assigning points outside boundaries to the nearest region

    Parameters:
        lat (float): Latitude
        lon (float): Longitude
        position_boundaries (dict, optional): Position boundary dictionary
    Returns:
        str: Position identifier 'P1', 'P2' or 'P3'
    """
    # Use default value if position boundaries are not provided
    if position_boundaries is None:
        return 'unknown'

    # First check if the point is in any region
    for pos, bounds in position_boundaries.items():
        if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                bounds['lon_min'] <= lon <= bounds['lon_max']):
            return pos

    # If not in any region, find the nearest region
    min_distance = float('inf')
    nearest_pos = 'unknown'

    for pos, bounds in position_boundaries.items():
        # Calculate minimum distance from point to region
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

        # Use Euclidean distance
        distance = (lat_dist ** 2 + lon_dist ** 2) ** 0.5

        if distance < min_distance:
            min_distance = distance
            nearest_pos = pos

    return nearest_pos


# Lane identification function remains unchanged
def identify_lane(lat, lon, bearing):
    """
    Identify lane (L1 or L2) based on latitude, longitude and bearing

    Parameters:
        lat (float): Latitude
        lon (float): Longitude
        bearing (float): Bearing angle
    Returns:
        str: Lane identifier 'L1' or 'L2'
    """
    # Classification based on bearing angle
    if 0 <= bearing < 180:
        return 'L1'
    else:
        return 'L2'


# Modified function for tracking position and lane changes
def track_position_lane_changes(df, position_boundaries=None):
    """
    Track position and lane changes

    Parameters:
        df (pd.DataFrame): Dataframe containing latitude, longitude, bearing
        position_boundaries (dict, optional): Position boundary dictionary
    Returns:
        pd.DataFrame: Dataframe with added position, lane and change identifiers
    """
    print("Identifying position and lane changes...")

    # Copy dataframe
    data = df.copy()

    # Initialize position and lane columns
    data['position'] = 'unknown'
    data['lane'] = 'unknown'
    data['position_change'] = 'no_change'
    data['lane_change'] = 'no_change'

    # Identify position and lane for each point
    for i in range(len(data)):
        if 'latitude' in data.columns and 'longitude' in data.columns:
            data.loc[i, 'position'] = identify_position(
                data.loc[i, 'latitude'],
                data.loc[i, 'longitude'],
                position_boundaries
            )

            # Use bearing angle (if available) to identify lane
            if 'bearing' in data.columns:
                data.loc[i, 'lane'] = identify_lane(
                    data.loc[i, 'latitude'],
                    data.loc[i, 'longitude'],
                    data.loc[i, 'bearing']
                )

    # Track position changes
    for i in range(1, len(data)):
        # Position change
        if data.loc[i, 'position'] != data.loc[i - 1, 'position']:
            if data.loc[i - 1, 'position'] != 'unknown' and data.loc[i, 'position'] != 'unknown':
                data.loc[i, 'position_change'] = f"{data.loc[i - 1, 'position']}→{data.loc[i, 'position']}"

        # Lane change
        if data.loc[i, 'lane'] != data.loc[i - 1, 'lane']:
            if data.loc[i - 1, 'lane'] != 'unknown' and data.loc[i, 'lane'] != 'unknown':
                data.loc[i, 'lane_change'] = f"{data.loc[i - 1, 'lane']}→{data.loc[i, 'lane']}"

    # Statistics of position and lane change distribution
    position_change_counts = data['position_change'].value_counts()
    lane_change_counts = data['lane_change'].value_counts()

    print("\nPosition Change Distribution:")
    for change, count in position_change_counts.items():
        if change != 'no_change':
            print(f"{change}: {count} ({count / len(data) * 100:.2f}%)")

    print("\nLane Change Distribution:")
    for change, count in lane_change_counts.items():
        if change != 'no_change':
            print(f"{change}: {count} ({count / len(data) * 100:.2f}%)")

    # New: Check if there are still unidentified points
    unknown_count = (data['position'] == 'unknown').sum()
    if unknown_count > 0:
        print(
            f"\nWarning: Still {unknown_count} points ({unknown_count / len(data) * 100:.2f}%) with unidentified position")

        # Optional: Save these points for further analysis
        unknown_points = data[data['position'] == 'unknown'][['latitude', 'longitude']]
        unknown_points.to_csv(os.path.join(results_dir, 'unknown_position_points.csv'), index=False)
        print(
            f"Coordinates of unidentified points saved to '{os.path.join(results_dir, 'unknown_position_points.csv')}'")

    return data


# Visualize position and lane changes function
def plot_position_lane_changes(df):
    """
    Plot distribution charts of position and lane changes

    Parameters:
        df (pd.DataFrame): Dataframe with position and lane change markers
    """
    # Position change pie chart
    position_changes = df[df['position_change'] != 'no_change']['position_change'].value_counts()
    if len(position_changes) > 0:
        plt.figure(figsize=(10, 8))
        plt.pie(position_changes, labels=position_changes.index, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.title('Position Change Distribution', fontsize=14)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'position_change_distribution.png'), dpi=300, bbox_inches='tight')

    # Lane change pie chart
    lane_changes = df[df['lane_change'] != 'no_change']['lane_change'].value_counts()
    if len(lane_changes) > 0:
        plt.figure(figsize=(10, 8))
        plt.pie(lane_changes, labels=lane_changes.index, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.title('Lane Change Distribution', fontsize=14)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'lane_change_distribution.png'), dpi=300, bbox_inches='tight')

    # Position and behavior relationship heatmap
    if 'behavior' in df.columns and 'position' in df.columns:
        plt.figure(figsize=(12, 8))

        # Calculate cross table of behavior and position
        cross_tab = pd.crosstab(df['position'], df['behavior'], normalize='index')

        # Plot heatmap
        sns.heatmap(cross_tab, annot=True, fmt='.2f', cmap='YlGnBu')
        plt.title('Behavior Distribution by Position', fontsize=14)
        plt.xlabel('Behavior', fontsize=12)
        plt.ylabel('Position', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'position_behavior_heatmap.png'), dpi=300, bbox_inches='tight')

    # New: Add position distribution chart
    plt.figure(figsize=(10, 6))
    position_counts = df['position'].value_counts()
    sns.barplot(x=position_counts.index, y=position_counts.values)
    plt.title('Position Distribution', fontsize=14)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'position_distribution.png'), dpi=300, bbox_inches='tight')


# Modified main function
def main(file_path, acc_threshold, decel_threshold, base_turn_threshold, use_gps=True):
    """
    Main function: Load data, train models, analyze results, and identify position and lane changes

    Parameters:
        file_path (str): Data file path
        acc_threshold (float): Acceleration threshold
        decel_threshold (float): Deceleration threshold
        base_turn_threshold (float): Base turning threshold
        use_gps (bool): Whether to use GPS data
    """
    print("===== Bicycle Motion Data Analysis and Behavior Recognition =====")
    print(f"Using GPS data: {'Yes' if use_gps else 'No'}")

    # 1. Load data
    df = load_data(file_path)

    position_boundaries = None

    # Only perform GPS-related analysis when using GPS data
    if use_gps and all(col in df.columns for col in ['latitude', 'longitude']):
        # New: Analyze GPS data range for setting position boundaries
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

        print(f"\nGPS Data Range:")
        print(f"Latitude range: {lat_min:.6f} - {lat_max:.6f}")
        print(f"Longitude range: {lon_min:.6f} - {lon_max:.6f}")

        # Modified: Use better boundary setting method with some overlap
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        # Add expansion factor for region overlap
        expansion_factor = 0.05  # 5% expansion

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

        print("\nSetting position boundaries (with 5% overlap):")
        for pos, bounds in position_boundaries.items():
            print(
                f"{pos}: Latitude {bounds['lat_min']:.6f}-{bounds['lat_max']:.6f}, Longitude {bounds['lon_min']:.6f}-{bounds['lon_max']:.6f}")
    else:
        if use_gps:
            print("Warning: Missing latitude/longitude information in data, position identification not possible")
        else:
            print("GPS data analysis disabled")

    # 2. Create features
    df_with_features = create_features(df, use_gps)

    # 3. Label behaviors
    df_labeled = label_behaviors(df_with_features, acc_threshold, decel_threshold, base_turn_threshold, use_gps)

    # Only perform position and lane analysis when using GPS data
    if use_gps:
        # 4. Mark position and lane changes (using set position boundaries)
        df_with_position_lane = track_position_lane_changes(df_labeled, position_boundaries)

        # 5. Visualize position and lane changes
        plot_position_lane_changes(df_with_position_lane)
    else:
        df_with_position_lane = df_labeled

    # 6. Save complete labeled data
    original_columns = df.columns.tolist()

    # If not using GPS data, don't include position and lane columns
    if use_gps:
        additional_columns = ['behavior', 'position', 'lane', 'position_change', 'lane_change']
    else:
        additional_columns = ['behavior']

    columns_to_save = original_columns + [col for col in additional_columns if col in df_with_position_lane.columns]
    complete_labeled_df = df_with_position_lane[columns_to_save]

    # Include whether GPS is used in the output file name
    file_suffix = "with_gps" if use_gps else "imu_only"
    output_file = os.path.join(results_dir, f"bike_data_labeled_{file_suffix}.xlsx")
    complete_labeled_df.to_excel(output_file, index=False)

    print(f"Complete labeled data saved as '{output_file}'")

    # 7. Select features
    X, y = select_features(df_labeled, use_gps)

    # 8. Train and evaluate models
    pipelines, results = train_and_evaluate_models(X, y)

    # 9. Save models
    for name, pipeline in pipelines.items():
        model_filename = os.path.join(results_dir, f"{name}_model_{file_suffix}.pkl")
        joblib.dump(pipeline, model_filename)
        print(f"Model saved as {model_filename}")

    # 10. Print summary report
    print("\n===== Model Performance Summary =====")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Training time: {result['train_time']:.2f} seconds")
        print(f"  Prediction time per sample: {result['predict_time_per_sample'] * 1000:.4f} milliseconds")

        # Print F1 score for each category
        report = result['classification_report']
        print("  F1 score by behavior:")
        for behavior in ['straight', 'acceleration', 'deceleration', 'turning']:
            if behavior in report:
                print(f"    - {behavior}: {report[behavior]['f1-score']:.4f}")

    print("\n===== Real-time Performance Analysis =====")
    # Sort models by prediction time
    sorted_models = sorted(results.items(), key=lambda x: x[1]['predict_time_per_sample'])

    print("Models sorted by prediction time per sample (from fastest to slowest):")
    for i, (name, result) in enumerate(sorted_models, 1):
        print(f"{i}. {name}: {result['predict_time_per_sample'] * 1000:.4f} ms/sample")

    # Determine best model (balancing accuracy and speed)
    best_model = None
    best_score = -1

    # Simple scoring function: accuracy / log(prediction time)
    for name, result in results.items():
        # Avoid taking log of 0
        pred_time = max(result['predict_time_per_sample'], 1e-6)
        score = result['accuracy'] / np.log10(pred_time * 1000 + 10)  # +10 for smoothing

        if score > best_score:
            best_score = score
            best_model = name

    print(f"\nConsidering balance of accuracy and speed, the best model is: {best_model}")

    return pipelines, complete_labeled_df, position_boundaries  # Return models, complete labeled data and position boundaries


# Modified prediction function with use_gps parameter
def predict_behavior(model, new_data, position_boundaries=None, use_gps=True):
    """
    Predict behavior for new data using trained model

    Parameters:
        model: Trained model pipeline
        new_data (pd.DataFrame): New sensor data
        position_boundaries (dict, optional): Position boundary dictionary
        use_gps (bool): Whether to use GPS data

    Returns:
        pd.DataFrame: Data with predicted behavior
    """
    # Preprocess new data
    processed_data = create_features(new_data, use_gps)

    # Get features used in training
    feature_names = model.feature_names_in_
    available_features = [f for f in feature_names if f in processed_data.columns]

    # Check if all required features are available
    if len(available_features) != len(feature_names):
        missing = set(feature_names) - set(available_features)
        print(f"Warning: Missing features in new data: {missing}")

    # Make prediction
    X_new = processed_data[available_features]
    predictions = model.predict(X_new)

    # Add predictions to dataframe
    processed_data['predicted_behavior'] = predictions

    # Add position and lane identification (only when use_gps=True)
    if use_gps and all(col in processed_data.columns for col in ['latitude', 'longitude']):
        processed_data_with_position = track_position_lane_changes(processed_data, position_boundaries)
        return processed_data_with_position
    else:
        return processed_data


# If this script is run directly
if __name__ == "__main__":
    # Set data file path
    file_path = r"E:\python_job\bike_analyze\bike\单车gps数据.xlsx"

    # Ask whether to use GPS analysis
    use_gps_input = input("Use GPS analysis? (yes/no): ").strip().lower()
    use_gps = use_gps_input in ['yes', 'y', 'true', '1']

    # Set behavior detection thresholds
    acc_threshold = 0.15
    decel_threshold = -0.15
    base_turn_threshold = 25

    # Run main function
    models, labeled_data, position_boundaries = main(file_path, acc_threshold, decel_threshold,
                                                     base_turn_threshold, use_gps)