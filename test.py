
import os
import time
import datetime
import shutil
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import backend as K

from utils import load_fold_data


def load_model(weights_path, model_id):
    json_file = weights_path + '/model' + str(model_id) + '.json'
    _file = open(json_file, 'r')
    model = model_from_json(_file.read())

    weight_file = weights_path + '/weights' + str(model_id) + '.hdf5'
    model.load_weights(weight_file)

    return model


def predict_data(folds, model, target_img_size, class_count, preprocess_input=None, binary_threshold=0.5):
    x_test = []
    y_test = []
    file_test = []
    start_time = datetime.datetime.now()
    print('start time: ', start_time)

    for fold in folds:
        _x_test, _y_test, _file_test = load_fold_data(fold, target_img_size)
        x_test += _x_test
        y_test += _y_test
        file_test += _file_test

    x_test = np.array(x_test)
    if preprocess_input:
        x_test = preprocess_input(x_test)

    predictions = model.predict(x_test, verbose=1)

    if class_count == 2:
        y_pred = [1 if predication[0] > binary_threshold else 0 for predication in predictions]
    else:
        y_pred = [np.argmax(predication) for predication in predictions]
    y_true = y_test

    err = 0
    for i in range(0, len(y_pred)):
        if (y_pred[i] != y_true[i]):
            err += 1

    acc = int((1 - err * 1.0 / len(y_pred)) * 100)

    end_time = datetime.datetime.now()
    cost_time = end_time - start_time
    print("end time {}, cost {}, fold: {} ,accurracy:{}".format(end_time, cost_time, folds, acc))

    return acc, y_pred, y_true, file_test, predictions


def extract_err_files(error_list, y_pred, y_true, other_fold, extract_all=False, remove=False):
    if os.path.exists(other_fold):
        shutil.rmtree(other_fold)

    move_files = 0
    for i in range(0, len(error_list)):
        if y_true[i] != y_pred[i] or extract_all:
            file = error_list[i]
            files = file.split('/')
            dirs = other_fold + '/' + str(y_true[i]) + '/' + str(y_pred[i]) + '/'
            if not os.path.exists(dirs):
                os.makedirs(dirs)

            dest = dirs + files[-1]

            if os.path.exists(file):
                move_files += 1
                shutil.copyfile(file, dest)
                if remove:
                    os.remove(file)

    print('extract {} files to {} finished'.format(move_files, other_fold))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def show_confuse_matrix(y_pred, y_true, classes_count, out_put_dir, fig_size=4, dpi=110, savefig=True):
    cnf_matrix = confusion_matrix(y_true, y_pred)

    # Plot non-normalized confusion matrix

    plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    classes = [str(x) for x in range(classes_count)]
    plot_confusion_matrix(cnf_matrix, classes=classes, title='Confusion matrix')

    rand = random.randint(1, 1000)
    file_name = time.strftime("%Y-%m-%d-%H-%M_", time.localtime()) + str(rand) + ".png"
    file_path = os.path.abspath(out_put_dir + '/' + file_name)
    if savefig:
        plt.savefig(file_path)
        print("save plot image to %s " % file_path)

    err = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] != y_true[i]:
            err += 1

    overall_acc = 1 - err * 1.0 / len(y_pred)
    print(cnf_matrix)
    acc_list = []
    for i in range(cnf_matrix.shape[0]):
        acc = 100 * cnf_matrix[i, i] / np.sum(cnf_matrix[i, :])
        print('%02d acc: %.2f%%' % (i, acc))
        acc_list.append(acc)
    print('overall acc: %.2f%%, avg acc: %.2f%%' % (100 * overall_acc, np.mean(acc_list)))

    return file_path


def get_roc(y_true, predicts, to_check_path_result, threshold_num=20, to_print=True):
    pred_list = []
    for itm in predicts:
        pred_list.extend(list(itm))

    data = list(zip(pred_list, y_true))
    thresholds = [i / threshold_num for i in range(0, threshold_num, 1)]
    thresholds.append(1)

    tp = []
    fp = []
    fn = []
    tn = []
    tpr = []
    tnr = []
    fpr = []
    for thrd in thresholds:
        thrd_tp, thrd_fp, thrd_fn, thrd_tn = [0] * 4
        for item in data:
            if item[1] == 1:
                if item[0] > thrd:
                    thrd_tp += 1
                else:
                    thrd_fn += 1
            elif item[1] == 0:
                if item[0] > thrd:
                    thrd_fp += 1
                else:
                    thrd_tn += 1

        thrd_tpr = round(float(thrd_tp) / (thrd_tp + thrd_fn), 3)
        thrd_fpr = round(float(thrd_fp) / (thrd_tn + thrd_fp), 3)

        tp.append(thrd_tp)
        fp.append(thrd_fp)
        fn.append(thrd_fn)
        tn.append(thrd_tn)
        tpr.append(thrd_tpr)
        tnr.append(round(1-thrd_fpr, 3))
        fpr.append(thrd_fpr)

    diff = [round(tpr[i] - fpr[i], 3) for i in range(len(tpr))]
    optimal_idx = np.argmax(diff)
    optimal_threshold = thresholds[optimal_idx]

    optimal_acc0 = round(tn[optimal_idx] / (tn[optimal_idx] + fp[optimal_idx]), 3) * 100
    optimal_acc1 = round(tp[optimal_idx] / (tp[optimal_idx] + fn[optimal_idx]), 3) * 100
    optimal_avg_acc = np.mean([optimal_acc0, optimal_acc1])
    optimal_overall_acc = round((tn[optimal_idx] + tp[optimal_idx]) / len(y_true), 3) * 100

    print('optimal_threshold: ', optimal_threshold, ' overall acc:  %.2f%%, avg acc: %.2f%%' % (optimal_overall_acc, optimal_avg_acc))

    if to_print:
        print("{}\t{}\t{}\t{}\t{}".format('thred', 'tpr', 'tnr', 'fpr', 'diff'))
        for i, thrd in enumerate(thresholds):
            print('{}\t{}\t{}\t{}\t{}'.format(thresholds[i], tpr[i], tnr[i], fpr[i], diff[i]))

    acc0 = [round(item / (item + fp[i]), 3) * 100 for i, item in enumerate(tn)]
    acc1 = [round(item / (item + fn[i]), 3) * 100 for i, item in enumerate(tp)]

    df = pd.DataFrame({'thresholds': thresholds, 'tpr': tpr, 'tnr': tnr, 'fpr': fpr, 'tpr-fpr': diff,
                       'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                       'acc0': acc0, 'acc1': acc1})
    df = df.ix[:, ['thresholds', 'tpr', 'tnr', 'fpr', 'tpr-fpr', 'tn', 'fp', 'fn', 'tp', 'acc0', 'acc1']]
    df.to_csv(to_check_path_result + r'\roc_%s_%s.csv' % (optimal_threshold, optimal_avg_acc), encoding='utf-8')


    fontsize = 14
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2)

    plt.ylabel('sensitivity', fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.xlabel('1-specificity', fontdict={'family': 'Times New Roman', 'size': fontsize})

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    new_ticks = np.linspace(0, 1, 11)
    plt.xticks(new_ticks, fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(new_ticks, fontproperties='Times New Roman', fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)

    plt.annotate(r'threshold={:.3f}'.format(optimal_threshold), xy=(fpr[optimal_idx], tpr[optimal_idx]),
                 xycoords='data', xytext=(+30, -30),
                 textcoords='offset points', fontsize=fontsize, color='blue',
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1", color='red'))

    plt.savefig(to_check_path_result + r'\roc_%s_%s.png' % (optimal_threshold, optimal_avg_acc))


def test_fold(to_chk_path, model, target_img_size, class_count, copy_err_files, preprocess_input, binary_threshold=0.5):
    to_check_path_result = to_chk_path + '_result_' + time.strftime("%y%m%d_%H%M%S", time.localtime(time.time()))
    os.makedirs(to_check_path_result, exist_ok=True)

    acc, y_pred, y_true, file_test, predicts = predict_data([to_chk_path], model,
                                                            target_img_size, class_count, preprocess_input,
                                                            binary_threshold=binary_threshold)

    if copy_err_files:
        extract_err_files(file_test, y_pred, y_true, to_check_path_result, extract_all=False)

    show_confuse_matrix(y_pred, y_true, class_count, to_check_path_result)

    if class_count == 2:
        get_roc(y_true, predicts, to_check_path_result)

    plt.show()


# configs
root_path = r'F:\bbps\data_root'
test_path = root_path + '/test'

model_name = 'resnet'
weights_path = root_path + '/cache'

img_size = (224, 224)
model_id = 1

class_count = 2
copy_err_files = True


#
if __name__ == '__main__':
    K.set_learning_phase(0)
    model = load_model(weights_path, model_id)

    preprocess_input = None
    if model_name == 'densenet':
        preprocess_input = densenet_preprocess_input
    elif model_name == 'xception':
        preprocess_input = xception_preprocess_input

    test_fold(test_path, model, img_size, class_count, copy_err_files, preprocess_input)

