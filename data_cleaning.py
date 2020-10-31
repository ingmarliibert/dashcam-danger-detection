from pathlib import Path
import pathlib
import pandas as pd
import shutil

TRAFFIC_SIGN_ROOT = './data/traffic-sign'
TRAIN_ROOT = './data/traffic-sign/train'
TEST_CLEAN = './data/traffic-sign/test-clean'


data_dir = pathlib.Path(TRAIN_ROOT)

def reorganize_test_pictures():
    """
    The dataset contains Test images without folder structure with label.
    e.g: 0 (folder) -> file_0.png, ...
    We need this folder structure to use some tensorflow helpers to load images.
    So we move test files accordingly.
    """
    test_df = pd.read_csv(f'{TRAFFIC_SIGN_ROOT}/Test.csv')
    test_df['Path'] = test_df['Path'].apply(lambda x: f'{TRAFFIC_SIGN_ROOT}/{x}').tolist()

    Path(TEST_CLEAN).mkdir(exist_ok=True)

    for i, test_row in test_df.iterrows():
        new_path = Path(f"{TEST_CLEAN}/{test_row['ClassId']}")
        new_path.mkdir(exist_ok=True)

        old_path = Path(test_row['Path'])

        shutil.copy(old_path, new_path)  # For newer Python.

    print(test_df)

reorganize_test_pictures()
