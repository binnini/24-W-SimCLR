import sys
import os
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Transformations import transformations

def read_csv(filepath):
    data = {}
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:
            class_index = int(row[0])
            class_name = row[1]
            accuracy = float(row[2])
            if class_index not in data:
                data[class_index] = {'class_name': class_name, 'accuracies': []}
            data[class_index]['accuracies'].append(accuracy)
    return data

def merge_csv(filepaths, method_names, output_file):
    merged_data = {}
    for filepath in filepaths:
        data = read_csv(filepath)
        for class_index, class_info in data.items():
            if class_index not in merged_data:
                merged_data[class_index] = {'class_name': class_info['class_name'], 'accuracies': []}
            merged_data[class_index]['accuracies'].extend(class_info['accuracies'])

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['Class Index', 'Class Name'] + [f'{name}_accuracy' for name in method_names]
        writer.writerow(headers)

        for class_index, class_info in merged_data.items():
            row = [class_index, class_info['class_name']] + class_info['accuracies']
            writer.writerow(row)

    print(f"Merged results saved to {output_file}")

if __name__ == "__main__":
    base_path = './results/dataAug/'
    output_file = os.path.join(base_path, 'model_results.csv')
    input_names = [name for name in transformations.keys() if name != 'original']
    input_files = []

    valid_input_names = []
    for name in input_names:
        file_path = os.path.join(base_path+'/per_method', f'cifar100_results_{name}.csv')
        if os.path.exists(file_path):
            input_files.append(file_path)
            valid_input_names.append(name)
        else:
            print(f"File not found: {file_path}. Skipping.")

    merge_csv(input_files, valid_input_names, output_file)