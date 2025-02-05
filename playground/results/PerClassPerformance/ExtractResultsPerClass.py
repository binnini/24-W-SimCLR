import pandas as pd
import os

# CSV 파일 경로
merged_results_path = '/hdd1/yebin/24-W-SimCLR/playground/results/merged_results.csv'
output_dir = '/hdd1/yebin/24-W-SimCLR/playground/results/PerClassPerformance'

# CSV 파일 읽기
df = pd.read_csv(merged_results_path)

# Augmentation 방법 목록
augmentations = ['crop', 'flip', 'color_jitter', 'grayscale', 'translation', 'shearing', 'rotation', 'center_mask', 'noise_injection', 'kernel_filtering', 'random_erasing']

# 각 클래스에 대해 결과 추출 및 저장
for class_index in range(100):
    class_data = df[df['Class Index'] == class_index]
    class_name = class_data['Class Name'].values[0]
    
    # 결과 데이터프레임 생성
    results = {
        'Augmentation': augmentations,
        'Original Performance': [class_data['original'].values[0]] * len(augmentations),
        'Augmentation Performance': [class_data[aug].values[0] for aug in augmentations],
        'Augmented Augmentation Performance': [class_data[f'aug_{aug}'].values[0] for aug in augmentations]
    }
    results_df = pd.DataFrame(results)
    
    # 파일 저장 경로
    output_path = os.path.join(output_dir, f'class_index_{class_index}_augmentations.csv')
    
    # CSV 파일로 저장
    results_df.to_csv(output_path, index=False)
    print(f"Saved results for class index {class_index} to {output_path}")