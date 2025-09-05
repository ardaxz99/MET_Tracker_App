import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, StratifiedKFold


# Activity → Intensity mapping
ACTIVITY_TO_INTENSITY = {
    "Sitting": "Sedentary",
    "Standing": "Sedentary",
    "Walking": "Light",
    "Upstairs": "Moderate",
    "Downstairs": "Moderate",
    "Jogging": "Vigorous"
}

INTENSITY_TO_ID = {
    "Sedentary": 0,
    "Light": 1,
    "Moderate": 2,
    "Vigorous": 3
}


class WISDMDataset(Dataset):
    def __init__(self, dataset_path, window_size_sec=5, sampling_rate=20,
                 stride_sec=1, k_fold=0, n_splits=10, train=True):
        """
        Args:
            dataset_path (str): Path to folder containing WISDM_ar_v1.1_raw.txt
            window_size_sec (int): Size of each window in seconds (default: 5)
            sampling_rate (int): Sampling rate (Hz)
            stride_sec (int|float): Stride between windows in seconds (default: 1)
            k_fold (int): Which fold to use as test set (0-9)
            n_splits (int): Total folds
            train (bool): Whether to return train or test split
        """
        self.window_size = int(window_size_sec * sampling_rate)
        self.step_size = int(stride_sec * sampling_rate)
        self.k_fold = k_fold
        self.n_splits = n_splits
        self.train = train

        # Load raw data
        raw_file = os.path.join(dataset_path, "WISDM_ar_v1.1_raw.txt")
        self.data = self._load_raw(raw_file)

        # Generate windows with time ordering
        X, y = self._create_windows()

        # Cross-validation split
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))

        train_idx, test_idx = splits[k_fold]
        if train:
            self.X = X[train_idx]
            self.y = y[train_idx]
        else:
            self.X = X[test_idx]
            self.y = y[test_idx]

    def _load_raw(self, file_path):
        """Load and clean WISDM raw file into DataFrame"""
        rows = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip().strip(";")  # remove spaces + trailing semicolon
                parts = line.split(",")
                if len(parts) != 6:
                    continue  # skip malformed rows

                user, activity, ts, x, y, z = parts

                # Skip if timestamp is zero or all accel values are zero
                if ts == "0" or (x == "0" and y == "0" and z == "0.0"):
                    continue

                rows.append(parts)

        df = pd.DataFrame(rows, columns=["user", "activity", "timestamp", "x", "y", "z"])

        # Convert numeric fields
        df["user"] = df["user"].astype(int)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df["z"] = pd.to_numeric(df["z"], errors="coerce")

        # Drop any rows with NaN (from bad parses)
        df = df.dropna().reset_index(drop=True)

        # Sort by user + timestamp
        df = df.sort_values(by=["user", "timestamp"]).reset_index(drop=True)

        return df


    def _create_windows(self):
        """Sequential windowing based on time"""
        X, y = [], []

        for user, user_df in self.data.groupby("user"):
            values = user_df[["x", "y", "z"]].values
            activities = user_df["activity"].values

            n_samples = len(values)

            start = 0
            while start + self.window_size <= n_samples:
                end = start + self.window_size
                window = values[start:end]
                window_activities = activities[start:end]

                # Assign window label = majority activity
                activity = pd.Series(window_activities).mode()[0]
                intensity = ACTIVITY_TO_INTENSITY[activity]
                label = INTENSITY_TO_ID[intensity]

                X.append(window)
                y.append(label)

                start += self.step_size  # move with stride

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
