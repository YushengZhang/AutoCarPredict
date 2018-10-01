import json
import os
import os.path as osp
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Pre-processing of the json files for AutoDrive')
    parser.add_argument('--rootdir', type=str,
                        default='/home/yusheng/PycharmProjects/AutoCarPredict/data/test_data_pre',
                        help="path to data root directory")
    parser.add_argument('--dataset', type=str,
                        default='track 20',
                        help="path to data root directory")
    parser.add_argument('--labeldir', type=str,
                        default='/home/yusheng/home/test_data',
                        help="path to data root directory")
    # parser.add_argument('--phase', type=str, default='train',
    #                     choices=['train', 'test'])
    args = parser.parse_args()

    datapath = osp.join(args.rootdir, args.dataset, 'log2')
    labelpath = osp.join(args.labeldir, args.dataset, 'log2')
    files = os.listdir(datapath)
    files = sorted(files, key=lambda s: int(s.split('_')[0]))
    for file in files:
        filetype = osp.splitext(file)[1]
        if filetype != '.json':
            continue
        filename = osp.splitext(file)[0]
        file_id = str(filename.split('_')[0])
        print(file_id)
        with open(osp.join(datapath, file), 'r') as s_json:
            load_dict = json.load(s_json)
            data = []
            label = []
            data.append(load_dict['right_wh_cross_vtx_norm'][0][0])
            data.append(load_dict['right_wh_cross_vtx_norm'][1][1])
            data.append(load_dict['right_wh_cross_vtx_norm'][2])
            data.append(load_dict['left_wh_cross_vtx_norm'][0][0])
            data.append(load_dict['left_wh_cross_vtx_norm'][1][1])
            data.append(load_dict['left_wh_cross_vtx_norm'][2])
        with open(osp.join(labelpath, '20_record_'+file_id+'.json'), 'r') as l_json:
            label_dict = json.load(l_json)
            data.append(label_dict['user/throttle'])
            label.append(label_dict['user/angle'])
        with open(osp.join(args.rootdir, 'test_data.json'), 'a+') as t_json:
            json_data = {"data": data, "label": label, "filename": args.dataset+'_'+file_id}
            json_data = json.dumps(json_data)
            t_json.write(json_data+'\n')
    print('Writing {} ends'.format(args.dataset))


if __name__ == '__main__':
    main()
