import os
import pickle

def convert_pickles_to_txt(exp_n):
    pickles_dir = f"runs/predict/exp{exp_n}/pickles"
    annotation_dir = f"runs/predict/exp{exp_n}/annotation"

    # Ensure annotation directory exists
    os.makedirs(annotation_dir, exist_ok=True)

    # Iterate over pickle files
    for filename in sorted(os.listdir(pickles_dir)):
        if filename.endswith(".pickle"):
            pickle_path = os.path.join(pickles_dir, filename)
            txt_filename = filename.replace(".pickle", ".txt")
            txt_path = os.path.join(annotation_dir, txt_filename)

            # Read pickle file
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            
            # Convert data to string and save as text
            with open(txt_path, "w") as f:
                for obj in data:
                    if obj.category.name == 'bird':
                        bbox = obj.bbox
                        score = obj.score.value  # Assuming score has a 'value' attribute
                        f.write(f"{bbox.minx}, {bbox.miny}, {bbox.maxx}, {bbox.maxy}, {score}\n")

            print(f"Converted {filename} to {txt_filename}")

# Example usage:
for i in range(12, 17):
    convert_pickles_to_txt(i)  # Change 1 to the desired exp{n} number
