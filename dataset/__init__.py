import pandas as pd
import os


def get_dataset(config, split_name):
    if config.data.dataset == "papila":
        # Read csv file
        if split_name == "test":
            split = pd.read_csv(os.path.join(config.data.data_dir, 'splits/', str(config.data.test_split)))
        elif split_name == "train":  # in-context learning exemplars
            split = pd.read_csv(os.path.join(config.data.data_dir, 'splits/', str(config.data.train_split)))
        else:
            raise ValueError
        # Here we will use image file name (xxxx.jpg) as the unique identifier
        # For each row, we create a dictionary
        dict_list = []
        for index, row in split.iterrows():
            dict_list.append({
                "image_id": row[config.data.image_name_col],
                "image_path": os.path.join(config.data.data_dir, "images/", row[config.data.image_name_col]),
                "gt_answer": row[config.data.diagnosis_col],
                "age": row[config.data.patient_age_col],
                # Binary age
                "b_age": 1 if row[config.data.patient_age_col] >= 60 else 0,
                "sex": row[config.data.patient_sex_col]})
        # Sort the list of dictionaries by image_id
        dict_list = sorted(dict_list, key=lambda x: x["image_id"])
        return dict_list
    # Modify below or add your own dataset
    elif config.data.dataset == "ham10000":
        # Read csv file
        split = pd.read_csv(os.path.join(config.data.data_dir, "split/my_test_age.csv"))
        # Here we will use image file name without ext. (xxxx) as the unique identifier
        # We select the "image_id", "Path", "binaryLabel", "age", "Sex" columns
        # Note that "binaryLabel=1" means malignant, "Sex=1" means female
        # For each row, we create a dictionary
        dict_list = []
        for index, row in split.iterrows():
            dict_list.append({
                "image_id": row["image_id"],
                "image_path": os.path.join(config.data.data_dir, row["Path"]),
                "gt_answer": row["binaryLabel"],
                "age": row["age"],
                "b_age": 1 if row["age"] >= 60 else 0,
                "sex": row["Sex"]})
        # Sort the list of dictionaries by image_id
        dict_list = sorted(dict_list, key=lambda x: x["image_id"])
        return dict_list
    elif config.data.dataset == "mimic_cxr":
        # Read csv file
        split = pd.read_csv(os.path.join(config.data.data_dir, "split/my_test.csv"))
        dict_list = []
        for index, row in split.iterrows():
            path = str(row["path_preproc"])
            # Find str that between the last occurrence of '/' and '.jpg'
            image_id = path.split('/')[-1].split('.')[0]
            image_name = image_id + '.jpg'
            dict_list.append({
                "image_id": image_id,
                "image_path": os.path.join(config.data.data_dir, "images", image_name),
                "gt_answer": 1 if row["disease_label"] == 1 else 0,
                "age": float(row["age"]),
                "b_age": 1 if row["age"] >= 60.0 else 0,
                "sex": int(row["sex_label"])})
        # Sort the list of dictionaries by image_id
        dict_list = sorted(dict_list, key=lambda x: x["image_id"])
        return dict_list
    else:
        raise ValueError(f"Dataset {config.data.dataset} not supported")
