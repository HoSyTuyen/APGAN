import os, argparse, networks
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import cv2
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--nb', type=int, default=4, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--pre_trained_model', required=True, default='pre_trained_model', help='pre_trained cartoongan model path')
parser.add_argument('--image_dir', required=True, default='image_dir', help='test image path')
parser.add_argument('--output_image_dir', required=False, default='output_image_dir', help='output test image path')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

G = networks.generator(args.in_ngc, args.out_ngc, args.ngf, args.nb)
if torch.cuda.is_available():
    G.load_state_dict(torch.load(args.pre_trained_model), strict = False)
else:
    # cpu mode
    G.load_state_dict(torch.load(args.pre_trained_model, map_location=lambda storage, loc: storage))
G.to(device)

src_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


name_list = os.listdir(args.image_dir)
name_list = [f for f in name_list if ('.jpg' in f) or ('.png' in f)]

if not os.path.exists(args.output_image_dir):
    os.mkdir(args.output_image_dir)

for name in tqdm(name_list):
    load_path = os.path.join(args.image_dir, name)
    save_out_path = os.path.join(args.output_image_dir, name)

    raw_image = cv2.imread(load_path)[:,:,::-1]

    raw_image = Image.fromarray(raw_image)
    x = src_transform(raw_image).to(device)
    x = torch.unsqueeze(x, 0)
    G_recon = G(x)[0]
    plt.imsave(save_out_path[:-3] + 'png', (G_recon.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)
