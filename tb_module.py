import tensorflow as tf
import datetime

def summary_writer(folder_name):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = folder_name+"/"+current_time
    return tf.summary.create_file_writer(train_log_dir)
