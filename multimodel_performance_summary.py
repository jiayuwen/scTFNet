from utils.metrics_utils import *
from models import *
import pandas as pd 

multi_model = tf.keras.models.load_model("./output/multi_2.5.fimo_labels_trans_unet_output/model/", custom_objects={'dice_coef':dice_coef, 'f1_metrics':f1_metrics,'matthews_correlation_coefficient':matthews_correlation_coefficient})

tf_list = ["Ase", "Dpax2", "Fer1", "gtSuH", "Hairless", "Insv", "Scute", "Sens", 'Seq', "Ttk69", "wtElba1", "wtElba2", "wtElba3", "wtInsv"]

header = ['the_tf','model','auc','pr_auc','f1_metrics','dice_coef','matthews_correlation_coefficient','binary_io_u']
final_df = pd.DataFrame(columns=header)

for idx, the_tf in enumerate(tf_list):
    print(the_tf)
    print("======================================================")
    tmp_model = tf.keras.models.load_model("./output/"+the_tf+"_2.5.fimo_labels_trans_unet_output/model/", custom_objects={'dice_coef':dice_coef, 'f1_metrics':f1_metrics,'matthews_correlation_coefficient':matthews_correlation_coefficient})
    test_dataset = tf.data.experimental.load("preprocessed_data/preprocessed_"+the_tf+"_2.5.fimo_labels_fimo_data/test_data")
    # tmp_model.evaluate(test_dataset.batch(100))
    
    dataset = test_dataset.enumerate()
    tmp_input = []
    tmp_label = []
    for element in dataset.as_numpy_iterator():
        tmp_input.append(element[1][0])
        tmp_label.append(element[1][1])
    
    single_label = tmp_model.predict(np.array(tmp_input), workers=12)
    tmp_dic={
        'auc': [tf.keras.metrics.AUC(curve="ROC", num_thresholds=1001, name="auc_roc")(tf.cast(tmp_label, tf.float32), single_label).numpy()],
        'pr_auc': [tf.keras.metrics.AUC(curve="PR", num_thresholds=1001, name="pr_auc")(tf.cast(tmp_label, tf.float32), single_label).numpy()],
        'f1_metrics': [f1_metrics(tf.cast(tmp_label, tf.float32), single_label).numpy()],
        'dice_coef': [dice_coef(tf.cast(tmp_label, tf.float32), single_label).numpy()],
        'matthews_correlation_coefficient': [matthews_correlation_coefficient(tf.cast(tmp_label, tf.float32), single_label).numpy()],
        'binary_io_u': [tf.keras.metrics.BinaryIoU([1], threshold=0.5)(tf.cast(tmp_label, tf.float32), single_label).numpy()]}

    tmp_dic['the_tf'] = the_tf
    tmp_dic['model'] = 'single_trans_unet'
    tmp_df = pd.DataFrame.from_dict(tmp_dic)
    final_df = pd.concat([final_df,tmp_df], ignore_index=True)
    print(tmp_dic)
    
    tf_idx = idx
    multi_label = multi_model.predict(np.array(tmp_input), workers=12)
    multi_tmp_label = np.expand_dims(multi_label[:,:,tf_idx], axis=-1)

    tmp_dic={
        'auc': [tf.keras.metrics.AUC(curve="ROC", num_thresholds=1001, name="auc_roc")(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()],
        'pr_auc': [tf.keras.metrics.AUC(curve="PR", num_thresholds=1001, name="pr_auc")(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()],
        'f1_metrics': [f1_metrics(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()],
        'dice_coef': [dice_coef(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()],
        'matthews_correlation_coefficient': [matthews_correlation_coefficient(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()],
        'binary_io_u': [tf.keras.metrics.BinaryIoU([1], threshold=0.5)(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()]}
    
    tmp_dic['the_tf'] = the_tf
    tmp_dic['model'] = 'multi_trans_unet'
    tmp_df = pd.DataFrame.from_dict(tmp_dic)
    final_df = pd.concat([final_df,tmp_df], ignore_index=True)
    print(tmp_dic)
    print("======================================================")

final_df.to_csv('./performance_summary/multi_model_2.5_summary.csv',index=False)

multi_model = tf.keras.models.load_model("./output/multi_6.5.fimo_labels_trans_unet_output/model/", custom_objects={'dice_coef':dice_coef, 'f1_metrics':f1_metrics,'matthews_correlation_coefficient':matthews_correlation_coefficient})

tf_list = ["Ase", "Chn", "Fer1", "gtSuH", "Ham", "Insv", "Scrt", "Sens", "Ttk"]

header = ['the_tf','model','auc','pr_auc','f1_metrics','dice_coef','matthews_correlation_coefficient','binary_io_u']
final_df = pd.DataFrame(columns=header)

for idx, the_tf in enumerate(tf_list):
    print(the_tf)
    print("======================================================")
    tmp_model = tf.keras.models.load_model("./output/"+the_tf+"_6.5.fimo_labels_trans_unet_output/model/", custom_objects={'dice_coef':dice_coef, 'f1_metrics':f1_metrics,'matthews_correlation_coefficient':matthews_correlation_coefficient})
    test_dataset = tf.data.experimental.load("preprocessed_data/preprocessed_"+the_tf+"_6.5.fimo_labels_fimo_data/test_data")
    # tmp_model.evaluate(test_dataset.batch(100))
    
    dataset = test_dataset.enumerate()
    tmp_input = []
    tmp_label = []
    for element in dataset.as_numpy_iterator():
        tmp_input.append(element[1][0])
        tmp_label.append(element[1][1])
    
    single_label = tmp_model.predict(np.array(tmp_input), workers=12)
    tmp_dic={
        'auc': [tf.keras.metrics.AUC(curve="ROC", num_thresholds=1001, name="auc_roc")(tf.cast(tmp_label, tf.float32), single_label).numpy()],
        'pr_auc': [tf.keras.metrics.AUC(curve="PR", num_thresholds=1001, name="pr_auc")(tf.cast(tmp_label, tf.float32), single_label).numpy()],
        'f1_metrics': [f1_metrics(tf.cast(tmp_label, tf.float32), single_label).numpy()],
        'dice_coef': [dice_coef(tf.cast(tmp_label, tf.float32), single_label).numpy()],
        'matthews_correlation_coefficient': [matthews_correlation_coefficient(tf.cast(tmp_label, tf.float32), single_label).numpy()],
        'binary_io_u': [tf.keras.metrics.BinaryIoU([1], threshold=0.5)(tf.cast(tmp_label, tf.float32), single_label).numpy()]}

    tmp_dic['the_tf'] = the_tf
    tmp_dic['model'] = 'single_trans_unet'
    tmp_df = pd.DataFrame.from_dict(tmp_dic)
    final_df = pd.concat([final_df,tmp_df], ignore_index=True)
    print(tmp_dic)
    
    tf_idx = idx
    multi_label = multi_model.predict(np.array(tmp_input), workers=12)
    multi_tmp_label = np.expand_dims(multi_label[:,:,tf_idx], axis=-1)

    tmp_dic={
        'auc': [tf.keras.metrics.AUC(curve="ROC", num_thresholds=1001, name="auc_roc")(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()],
        'pr_auc': [tf.keras.metrics.AUC(curve="PR", num_thresholds=1001, name="pr_auc")(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()],
        'f1_metrics': [f1_metrics(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()],
        'dice_coef': [dice_coef(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()],
        'matthews_correlation_coefficient': [matthews_correlation_coefficient(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()],
        'binary_io_u': [tf.keras.metrics.BinaryIoU([1], threshold=0.5)(tf.cast(tmp_label, tf.float32), multi_tmp_label).numpy()]}
    
    tmp_dic['the_tf'] = the_tf
    tmp_dic['model'] = 'multi_trans_unet'
    tmp_df = pd.DataFrame.from_dict(tmp_dic)
    final_df = pd.concat([final_df,tmp_df], ignore_index=True)
    print(tmp_dic)
    print("======================================================")
    
final_df.to_csv('./performance_summary/multi_model_6.5_summary.csv',index=False)