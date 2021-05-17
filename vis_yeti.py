import json
import matplotlib.pyplot as plt

from datasets.oxford import get_dataloaders
from datasets.boreas import get_dataloaders_boreas
from networks.under_the_radar import UnderTheRadar
from networks.hero import HERO
from networks.yeti import YETI
from utils.utils import computeMedianError, computeKittiMetrics, saveKittiErrors, save_in_yeti_format, get_T_ba
from utils.utils import load_icra21_results, getStats, get_inverse_tf, get_folder_from_file_path
from utils.vis import convert_plt_to_img
import cv2

def draw_match(batch, out, config, solver, inliers):
    cart_pixel_width = config['cart_pixel_width']
    cart_resolution = config['cart_resolution']
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    T_met_pix = np.array([[0, -cart_resolution, 0, cart_min_range],
                          [cart_resolution, 0, 0, -cart_min_range],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    T_pix_met = np.linalg.inv(T_met_pix)
    keypoint_ints = out['keypoint_ints']
    ids = torch.nonzero(keypoint_ints[0, 0] > 0, as_tuple=False).squeeze(1)
    ids_cpu = ids.cpu()
    src = out['src'][0, ids].squeeze().detach().cpu().numpy()
    tgt = out['tgt'][0, ids].squeeze().detach().cpu().numpy()
    inliers = inliers[0]
    src = src[inliers]
    tgt = tgt[inliers]
    T_tgt_src = get_T_ba(out, a=0, b=1)
    plt.figure()
    plt.imshow(radar, cmap='gray', extent=(0, 640, 640, 0), interpolation='none')
    for i in range(src.shape[0]):
        x1 = np.array([src[i, 0], src[i, 1], 0, 1]).reshape(4, 1)
        x2 = np.array([tgt[i, 0], tgt[i, 1], 0, 1]).reshape(4, 1)
        x1 = T_tgt_src @ x1
        x1 = T_pix_met @ x1
        x2 = T_pix_met @ x2
        plt.plot([x1[0, 0], x2[0, 0]], [x1[1, 0], x2[1, 0]], c='w', linewidth=1, zorder=2)
        plt.scatter(x1[0, 0], x1[1, 0], c='limegreen', s=2, zorder=3)
        plt.scatter(x2[0, 0], x2[1, 0], c='r', s=2, zorder=4)
    plt.axis('off')
    return convert_plt_to_img()
    # plt.savefig('match1.png', bbox_inches='tight', pad_inches=0.0)

def get_closest_image(folder, ref_time):
    files = os.listdir(folder)
    min_delta = 1e6
    closest = -1
    for i, file in enumerate(files):
        time = int(file.split('.')[0])
        delta = abs(time - ref_time)
        if delta < min_delta:
            min_delta = delta
            closest = i
    assert(closest != -1)
    frame = cv2.imread(folder + files[closest])
    H = 720
    W = 1280
    upper_crop = 240
    h_before_resize = 1377
    frame_rate = 30
    frame = frame[upper_crop:upper_crop+h_before_resize]
    return cv2.resize(frame, (W, H))

def panel_and_save(match_img, cam_img, odom_img, save_path):
    match_img = cv2.resize(match_img, (640, 640))
    odom_img = cv2.resize(odom_img, (640, 640))
    comb = cv2.hconcat([match_img, odom_img])
    comb = cv2.vconcat([cam_img, comb])
    cv2.imsave(save_path)

if __name__ == '__main__':
    with open('config/steam.json') as f:
        config = json.load(f)
    root = './'

    if config['model'] == 'UnderTheRadar':
        model = UnderTheRadar(config).to(config['gpuid'])
    elif config['model'] == 'HERO':
        model = HERO(config).to(config['gpuid'])
        model.solver.sliding_flag = True
    elif config['model'] == 'YETI':
        model = YETI(config)
        model.solver.sliding_flag = True

    if config['model'] == 'UnderTheRadar' or config['model'] == 'HERO':
        assert(args.pretrain is not None)
        checkpoint = torch.load(args.pretrain, map_location=torch.device(config['gpuid']))
            failed = False
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except Exception as e:
            print(e)
            failed = True
        if failed:
            model.load_state_dict(checkpoint, strict=False)

    model.eval()

    if config['dataset'] == 'oxford':
        _, _, test_loader = get_dataloaders(config)
    elif config['dataset'] == 'boreas':
        _, _, test_loader = get_dataloaders_boreas(config)

    seq_name = test_loader.dataset.sequences[0]

    T_gt = []
    T_pred = []

    for batchi, batch in enumerate(test_loader):
        with torch.no_grad():
            out = model(batch)
        T_gt.append(batch['T_21'][w].numpy().squeeze())
        T_pred.append(get_T_ba(out, a=0, b=1))
        inliers = []
        model.solver.solver_cpp.getInliers(inliers)
        match_img = draw_match(batch, out, config, solver, inliers)
        print(match_img.shape)
        cv2.imsave('match.png', match_img)

        # Get closest camera image, crop it
        radar_time = batch['t_ref'][0][0].item()
        cam_img = get_closest_image(radar_time, config['data_dir'] + seq_name + '/camera/')
        print(cam_img.shape)
        cv2.imsave('cam.png', cam_img)


        # draw odom path 
        odom_img = plot_sequences(T_gt, T_pred, [len(T_gt)], returnTensor=False)
        print(odom_img.shape)
        cv2.imsave('odom.png', odom_img)

        # panel the images together, save as png file
        panel_and_save(match_img, cam_img, odom_img, root + str(batchi) + '.png')

        break


