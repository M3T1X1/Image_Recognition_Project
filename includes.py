import shutil
import kagglehub
import os
import json

data_path = 'data'

def generate_and_map_data(data_path):
    train_path = os.path.join(data_path, "train")

    classes = sorted(d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d)))

    mapping = {}

    for class_name in classes:
        name = class_name.lower()

        if 'suv' in name:
            category = "suv"
        elif 'coupe' in name:
            category = "coupe"
        elif 'sedan' in name:
            category = "sedan"
        elif 'hatchback' in name:
            category = "hatchback"
        elif 'convertible' in name:
            category = "convertible"
        elif 'wagon' in name:
            category = "wagon"
        else:
            category = "other"

        mapping[class_name] = category

    with open('mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)

    print(f" {len(mapping)} classess mapped.")


def download_and_copy():
    if not os.path.exists(data_path) or not os.listdir(data_path):
        source_path = kagglehub.dataset_download("jutrera/stanford-car-dataset-images-in-224x224")

        actual_data_dir = os.path.join(source_path, "stanford-car-dataset-by-classes-folder-224", "car_data")

        shutil.copytree(actual_data_dir, data_path, dirs_exist_ok=True)

        print(f"Done downloading and copying.")

    else:
        pass


download_and_copy()
generate_and_map_data(data_path)