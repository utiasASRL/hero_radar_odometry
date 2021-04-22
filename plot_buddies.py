import pickle
from utils.vis import plot_sequences
from utils.utils import load_icra21_results
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root1 = '/home/keenan/Desktop/logs/2021-03-04/boreas1_aug026_cnn8_window4_res2240_width640/'
    # root2 = '/home/keenan/Desktop/logs/2021-02-21/run45_aug026_cnn8_res2592_window3/'
    root2 = '/home/keenan/Desktop/logs/2021-04-02/run233_aug026_cnn8_res25_window4/'

    sequence = 'boreas-2021-01-26-10-59'

    T_gt, T_pred1, _ = pickle.load(open(root1 + 'odom' + sequence + '.obj', 'rb'))
    _, T_pred2, _ = pickle.load(open(root2 + 'odom' + sequence + '.obj', 'rb'))
    fname = './' + sequence + '.pdf'
    plot_sequences(T_gt, T_pred1, [len(T_gt)], returnTensor=False, savePDF=True, fnames=[fname],
                   T_icra=T_pred2, mainlabel='Trained on Custom', icralabel='Trained on Oxford', icracolor='limegreen',
                   legend_pos='upper left', rot=-90)
