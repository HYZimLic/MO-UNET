import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from unet.model import segmounet
from utils.data_vis import plot_img_and_mask
import re
import pandas as pd


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.3):
    net.eval()

    img = torch.from_numpy(full_img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        # if net.n_classes > 1:
        #     probs = F.softmax(output, dim=1)
        # else:
        #     probs = torch.sigmoid(output)
        probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize([full_img.shape[1], full_img.shape[2]]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask
    # return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--deepsupervision', default=0)
    parser.add_argument('--model', '-m',
                        default=r'G:\csvd\segmounet\checkpoints\CP_epoch70.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images')
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        default=r'G:\csvd\cmbpre/',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.1)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)
    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], '.png'))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255 ).astype(np.uint8))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    imagedata = pd.read_csv('./test.csv', header=None)
    in_files = imagedata.iloc[:, :].values
    out_files = args.output[0]
    # index value from npymax.py, each represents the maximum value for each lesion
    CMBindex = 12970.0
    WHMindex = 5061.0
    epvsindex = 9704.0
    lacuneindex = 5061.0
    if not os.path.exists(out_files):
        os.makedirs(out_files)
    net = segmounet(
        encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    for i, fn in enumerate(in_files):
        name = np.asarray(re.findall(r"\d+", fn[0]), np.chararray)

        logging.info("\nPredicting image {} ...".format(fn))
        img = np.load(fn[0])
        if img[0, 0] == 1:
            img = img / CMBindex
        elif img[0, 0] == 2:
            img = img/ WHMindex
        elif img[0, 0] == 3:
            img = img / epvsindex
        elif img[0, 0] == 4:
            img = img / lacuneindex
        img = np.expand_dims(img, 0)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files
            save_path = out_files + '/' + os.path.split(fn[0])[1].replace("npy", "png")
            result = mask_to_image(mask)
            result.save(save_path)

            logging.info("Mask saved to {}".format(out_files))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)


