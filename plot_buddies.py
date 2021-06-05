import pickle
from utils.vis import plot_sequences
from utils.utils import load_icra21_results
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root1 = '/home/keenan/Desktop/logs/2021-05-28/boreas1_res2384_window4/'
    # root2 = '/home/keenan/Desktop/logs/2021-02-21/run45_aug026_cnn8_res2592_window3/'
    root2 = '/home/keenan/Desktop/logs/2021-05-28/run342_res2592_window4/'

    sequence = 'boreas-2021-01-26-11-22'

    T_gt, T_pred1, _ = pickle.load(open(root1 + 'odom' + sequence + '.obj', 'rb'))
    _, T_pred2, _ = pickle.load(open(root2 + 'odom' + sequence + '.obj', 'rb'))
    fname = './' + sequence + '.pdf'
    plot_sequences(T_gt, T_pred1, [len(T_gt)], returnTensor=False, savePDF=True, fnames=[fname],
                   T_icra=T_pred2, mainlabel='Trained on Boreas', icralabel='Trained on Oxford', icracolor='limegreen',
                   legend_pos='upper left', rot=-90)
