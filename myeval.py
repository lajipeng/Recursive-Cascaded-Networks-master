import argparse
import os
import json
import re
import numpy as np
# python eval.py -c weights/VTN-10-liver -g YOUR_GPU_DEVICES
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default='weights/VTN-10-liver',
                    help='Specifies a previous checkpoint to load')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='-1',
                    help='Specifies gpu device(s)')
parser.add_argument('-d', '--dataset', type=str, default=None,
                    help='Specifies a data config')
parser.add_argument('-v', '--val_subset', type=str, default=None)
parser.add_argument('--batch', type=int, default=4, help='Size of minibatch')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--paired', action='store_true')
parser.add_argument('--data_args', type=str, default=None)
parser.add_argument('--net_args', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
import tflearn

import network
import data_util.liver
import data_util.brain


def main():
    if args.checkpoint is None:
        print('Checkpoint must be specified!')
        return
    if ':' in args.checkpoint:
        args.checkpoint, steps = args.checkpoint.split(':')
        steps = int(steps)
    else:
        steps = None
    args.checkpoint = find_checkpoint_step(args.checkpoint, steps)
    print(args.checkpoint)
    model_dir = os.path.dirname(args.checkpoint)
    try:
        with open(os.path.join(model_dir, 'args.json'), 'r') as f:
            model_args = json.load(f)
        print(model_args)
    except Exception as e:
        print(e)
        model_args = {}

    if args.dataset is None:
        args.dataset = model_args['dataset']
    if args.data_args is None:
        args.data_args = model_args['data_args']

    Framework = network.FrameworkUnsupervised
    Framework.net_args['base_network'] = model_args['base_network']
    Framework.net_args['n_cascades'] = model_args['n_cascades']
    Framework.net_args['rep'] = args.rep
    Framework.net_args.update(eval('dict({})'.format(model_args['net_args'])))
    if args.net_args is not None:
        Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    with open(os.path.join(args.dataset), 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type')
    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))
    framework = Framework(devices=gpus, image_size=image_size, segmentation_class_value=cfg.get(
        'segmentation_class_value', None), fast_reconstruction=args.fast_reconstruction, validation=True)
    print('Graph built')

    Dataset = eval('data_util.{}.Dataset'.format(image_type))
    ds = Dataset(args.dataset, batch_size=args.batch, paired=args.paired, **
                 eval('dict({})'.format(args.data_args)))

    sess = tf.Session()

    saver = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES))
    checkpoint = args.checkpoint
    saver.restore(sess, checkpoint)
    tflearn.is_training(False, session=sess)

    val_subsets = [data_util.liver.Split.VALID]
    if args.val_subset is not None:
        val_subsets = args.val_subset.split(',')

    tflearn.is_training(False, session=sess)
    if not os.path.exists('pair'):
        os.mkdir('pair')
    for val_subset in val_subsets:
        print("Validation subset {}".format(val_subset))
        gen = ds.generator(val_subset, loop=False)
        results = framework.my_validate(sess, gen, keys=None, summary=False)
        indexname = ['warped_moving','warped_moving0','warped_moving2','warped_moving3',\
        'warped_moving4','warped_moving5','warped_moving6','warped_moving7','warped_moving8',\
            'warped_moving9','warped_moving10','image_fixed']
        warped_moving =  results['warped_moving'][0][0][:,:,:,0]
        warped_moving0 = results['warped_moving_0'][0][0][:,:,:,0]
        warped_moving1 = results['warped_moving_1'][0][0][:,:,:,0]
        warped_moving2 = results['warped_moving_2'][0][0][:,:,:,0]
        warped_moving3 = results['warped_moving_3'][0][0][:,:,:,0]
        warped_moving4 = results['warped_moving_4'][0][0][:,:,:,0]
        warped_moving5 = results['warped_moving_5'][0][0][:,:,:,0]
        warped_moving6 = results['warped_moving_6'][0][0][:,:,:,0]
        warped_moving7 = results['warped_moving_7'][0][0][:,:,:,0]
        warped_moving8 = results['warped_moving_8'][0][0][:,:,:,0]
        warped_moving9 = results['warped_moving_9'][0][0][:,:,:,0]
        warped_moving10 = results['warped_moving_10'][0][0][:,:,:,0]
        image_fixed = results['image_fixed'][0][0][:,:,:,0]
        np.savez('array_save.npz',warped_moving,warped_moving0,warped_moving1,warped_moving2,warped_moving3,\
        warped_moving4,warped_moving5,warped_moving6,warped_moving7,warped_moving8,\
            warped_moving9,warped_moving10,image_fixed,indexname)






def find_checkpoint_step(checkpoint_path, target_steps=None):
    pattern = re.compile(r'model-(\d+).index')
    checkpoints = []
    for f in os.listdir(checkpoint_path):
        m = pattern.match(f)
        if m:
            steps = int(m.group(1))
            checkpoints.append((-steps if target_steps is None else abs(
                target_steps - steps), os.path.join(checkpoint_path, f.replace('.index', ''))))
    return min(checkpoints, key=lambda x: x[0])[1]


if __name__ == '__main__':
    main()
