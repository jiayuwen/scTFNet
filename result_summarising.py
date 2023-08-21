import csv
import pandas as pd 
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Result Summarising")
    parser.add_argument('-p', '--peak_calling_method', default='_6.5.fimo_labels', type=str,
                        help='peak calling method')
    parser.add_argument('-f', '--path', default='/g/data/ik06/stark/DeepTF_copy/', type=str,
                        help='save data to path')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)
    peak_method = args.peak_calling_method
    file_path = args.path
    output_dir = file_path + 'performance_summary/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if peak_method != '_6.5.fimo_labels':
        tf_list=["Ase", "Dpax2", "Fer1", "gtSuH", "Hairless", "Insv", "Scute", "Sens", "Seq", "Ttk69", "wtElba1", "wtElba2", "wtElba3", "wtInsv"]
    else:
        tf_list=["Ase", "Chn", "Fer1", "gtSuH", "Ham", "Insv", "Scrt", "Sens", "Ttk"]
        
    model_list=[ 'trans_unet', 'unet', 'se_unet'] #[ 'trans_unet', 'unet', 'se_cnn']

    header = ['the_tf','model','loss','auc','pr_auc','f1_metrics','dice_coef','matthews_correlation_coefficient','binary_io_u','training_time']
    final_df = pd.DataFrame(columns=header)

    for the_tf in tf_list:
        for model_arch in model_list:
            the_path = file_path + 'output/' + ''.join(the_tf) + peak_method + '_' + model_arch + "_output/result/evaluation_metrics.csv"
            tmp_df = pd.read_csv(the_path)
            tmp_dic = tmp_df.to_dict()
            tmp_dic['the_tf'] = the_tf
            tmp_dic['model'] = model_arch
            tmp_df = pd.DataFrame.from_dict(tmp_dic)
            final_df = pd.concat([final_df,tmp_df], ignore_index=True)
    
    print("generating performance summary for: " + peak_method[1:])
    final_df.to_csv(output_dir + peak_method[1:] +'summary.csv', index=False)

if __name__ == '__main__':
    main()