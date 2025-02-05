import pandas as pd

def merge_csv_files(basic_results_file, model_results_file, output_file):
    # Load the CSV files
    basic_df = pd.read_csv(basic_results_file)
    model_df = pd.read_csv(model_results_file)

    # Rename columns in model_df
    model_df = model_df.rename(columns=lambda x: f'aug_{x.replace("_accuracy", "")}' if '_accuracy' in x else x)

    # Merge the dataframes on 'Class Index' and 'Class Name'
    merged_df = pd.merge(basic_df, model_df, on=['Class Index', 'Class Name'])

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged results saved to {output_file}")

def add_sensitivity_performance(input_file, output_file):
    # CSV 파일 읽기
    df = pd.read_csv(input_file)

    # 'aug_'로 시작하는 열 이름에서 'aug_'를 제거한 방법론 목록 생성
    methods = [col.replace('aug_', '') for col in df.columns if col.startswith('aug_')]
    print(methods)
    # sensitivity와 performance enhancement 계산 및 추가
    for method in methods:
        df[f'{method}_sensitivity'] = df['original'] - df[f'{method}']
        df[f'{method}_performance_enhancement'] = df[f'aug_{method}'] - df['original']

    # 새로운 CSV 파일로 저장
    df.to_csv(output_file, index=False)
    print(f"New CSV file with sensitivity and performance enhancement saved to {output_file}")
    
if __name__ == "__main__":
    basic_results_file = './results/basic_model/cifar100_basic_results.csv'
    model_results_file = './results/dataAug/model_results.csv'
    output_file = './results/merged_results.csv'

    merge_csv_files(basic_results_file, model_results_file, output_file)
    add_sensitivity_performance(output_file, './results/merged_results_with_sensitivitiy&performance.csv')