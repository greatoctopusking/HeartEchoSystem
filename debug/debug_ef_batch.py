"""
批量处理数据集中所有patient，计算EF并与cfg文件中的EF比较
"""
import os
import sys
import numpy as np
import nibabel as nib
import configparser

sys.stdout.reconfigure(encoding='utf-8')

from biplane_simpson_clinical import BiplaneSimpsonClinical


BASE_PATH = r"D:\SRTP_Project__DeepLearning\project\Resources\database_nifti"
OUTPUT_FILE = r"D:\SRTP_Project__DeepLearning\HeartEchoSystem10\HeartEchoSystem\backend\debug_simpson_output\ef_comparison.csv"


def _ensure_int_labels(mask_arr):
    arr = np.asarray(mask_arr)
    if arr.dtype.kind in ("i", "u"):
        return arr
    return np.rint(arr).astype(np.int16)


def read_cfg_ef(patient_id):
    """从cfg文件中读取EF值"""
    cfg_path = os.path.join(BASE_PATH, patient_id, "Info_2CH.cfg")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(BASE_PATH, patient_id, "Info_4CH.cfg")
    
    if not os.path.exists(cfg_path):
        return None
    
    try:
        with open(cfg_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('EF:'):
                    ef = int(line.split(':')[1].strip())
                    return ef
    except:
        return None
    
    return None


def process_patient(patient_id):
    """处理单个病人的数据"""
    file_2ch = os.path.join(BASE_PATH, patient_id, f"{patient_id}_2CH_half_sequence_gt.nii.gz")
    file_4ch = os.path.join(BASE_PATH, patient_id, f"{patient_id}_4CH_half_sequence_gt.nii.gz")
    
    if not os.path.exists(file_2ch) or not os.path.exists(file_4ch):
        return None
    
    try:
        img2 = nib.load(file_2ch)
        img4 = nib.load(file_4ch)
        
        masks2 = img2.get_fdata()
        masks4 = img4.get_fdata()
        spacing2 = img2.header.get_zooms()[:2]
        spacing4 = img4.header.get_zooms()[:2]
        
        masks2_list = [masks2[:, :, t] for t in range(masks2.shape[2])]
        masks4_list = [masks4[:, :, t] for t in range(masks4.shape[2])]
        
        calculator = BiplaneSimpsonClinical(n_discs=20)
        
        result = calculator.compute_ed_es_from_series(
            masks2_list, masks4_list, spacing2, spacing4
        )
        
        return {
            'EDV': result['EDV'],
            'ESV': result['ESV'],
            'EF': result['EF'],
            'ED_index_2ch': result['ED_index'],
            'ES_index_2ch': result['ES_index'],
        }
    except Exception as e:
        print(f"  [ERROR] {patient_id}: {e}")
        return None


def get_all_patients():
    """获取所有patient文件夹"""
    patients = []
    for i in range(1, 501):
        patient_id = f"patient{i:04d}"
        patient_path = os.path.join(BASE_PATH, patient_id)
        if os.path.exists(patient_path):
            patients.append(patient_id)
    return patients


def main():
    patients = get_all_patients()
    print(f"Found {len(patients)} patients")
    
    results = []
    
    for idx, patient_id in enumerate(patients):
        print(f"[{idx+1}/{len(patients)}] Processing {patient_id}...", end=" ")
        
        # 读取cfg中的EF
        cfg_ef = read_cfg_ef(patient_id)
        
        # 计算EF
        calc_result = process_patient(patient_id)
        
        if calc_result is not None and cfg_ef is not None:
            calc_ef = calc_result['EF']
            error = abs(calc_ef - cfg_ef) / cfg_ef * 100
            
            results.append({
                'patient': patient_id,
                'cfg_ef': cfg_ef,
                'calc_ef': calc_ef,
                'error_pct': error,
                'EDV': calc_result['EDV'],
                'ESV': calc_result['ESV'],
            })
            print(f"CFG EF={cfg_ef}%, Calc EF={calc_ef:.2f}%, Error={error:.2f}%")
        elif calc_result is None:
            print(f"SKIP (calc failed)")
        else:
            print(f"SKIP (no cfg EF)")
    
    # 输出结果
    print(f"\n\n{'='*80}")
    print(f"EF COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Patient':<12} {'CFG_EF':<10} {'Calc_EF':<12} {'Error(%)':<12} {'EDV':<10} {'ESV':<10}")
    print("-" * 80)
    
    total_error = 0
    valid_count = 0
    
    for r in results:
        print(f"{r['patient']:<12} {r['cfg_ef']:<10} {r['calc_ef']:<12.2f} {r['error_pct']:<12.2f} {r['EDV']:<10.2f} {r['ESV']:<10.2f}")
        total_error += r['error_pct']
        valid_count += 1
    
    print("-" * 80)
    
    if valid_count > 0:
        avg_error = total_error / valid_count
        print(f"\nTotal: {valid_count} patients processed successfully")
        print(f"Average Absolute Relative Error: {avg_error:.2f}%")
    
    # 保存到CSV
    import csv
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient', 'CFG_EF', 'Calc_EF', 'Error_Pct', 'EDV', 'ESV'])
        for r in results:
            writer.writerow([r['patient'], r['cfg_ef'], f"{r['calc_ef']:.2f}", f"{r['error_pct']:.2f}", f"{r['EDV']:.2f}", f"{r['ESV']:.2f}"])
    
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
