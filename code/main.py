# CSCI 1430
# Homework 1 Image Filtering
#
import argparse
import os
from hw1_filtertest import filter_test
from hw1_fitkernel import recover_kernel
from hw1_hybridimage import hybrid_img_generation

def main():
    """
    Usage:
        python main.py -t filtertest -i ../data/dog.bmp
        python main.py -t fitkernel
        python main.py -t fitkernel -m ls
        python main.py -t hybridimage -i ../data/cat.bmp,../data/dog.bmp

    Arguments:
        -t, --task   Task to run: filtertest, fitkernel, or hybridimage
        -i, --image  Image path(s). For hybridimage, separate two paths with a comma
        -m, --mode   For fitkernel: 'gd' (gradient descent, default) or 'ls' (least squares)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', required=True,
                        choices=['filtertest', 'fitkernel', 'hybridimage'],
                        help='Task to run')
    parser.add_argument('-i', '--image',
                        help='Image path(s), comma-separated for hybridimage')
    parser.add_argument('-m', '--mode', choices=['gd', 'ls'], default='gd',
                        help="For fitkernel: 'gd' or 'ls' (default: gd)")
    args = parser.parse_args()

    if args.task == 'filtertest':
        if not os.path.exists(args.image):
            print(f'File not found: {args.image}')
        else:
            print(f'Running filter tests on {args.image}')
            filter_test(args.image)

    elif args.task == 'fitkernel':
        print(f'Running kernel fitting')
        recover_kernel(mode=args.mode)

    elif args.task == 'hybridimage':
        img1, img2 = args.image.split(',')
        if not os.path.exists(img1):
            print(f'File not found: {img1}')
        elif not os.path.exists(img2):
            print(f'File not found: {img2}')
        else:
            print(f'Running hybrid image generation on {img1} and {img2}')
            hybrid_img_generation(img1, img2)

if __name__ == '__main__':
    main()
