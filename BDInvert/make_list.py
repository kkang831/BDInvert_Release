import glob
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('image_list', type=str,
                    help='List of images to invert.')
parser.add_argument('--file_type', type=str, default='jpg', help='jpg or png?')
parser.add_argument('--save_name', type=str, default='test.list', help='Save file location ')
args = parser.parse_args()
import glob
image_list = []
for filename in glob.glob(f'{args.image_list}/*.{args.file_type}'):
    image_list.append(os.path.basename(filename))

with open(os.path.join(args.image_list,args.save_name), 'w') as f:
    for item in image_list:
        f.write(f"{os.path.join(args.image_list, item)}\n")
