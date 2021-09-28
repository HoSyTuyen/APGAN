import os, time, pickle, argparse, networks, utils
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import cv2
from PIL import Image

parser = argparse.ArgumentParser()
# parser.add_argument('--name', required=False, default='project_name',  help='')
# parser.add_argument('--src_data', required=False, default='src_data_path',  help='sec data path')
# parser.add_argument('--tgt_data', required=False, default='tgt_data_path',  help='tgt data path')
# parser.add_argument('--vgg_model', required=False, default='pre_trained_VGG19_model_path/vgg19.pth', help='pre-trained VGG19 model path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
# parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
# parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--ngf', type=int, default=32)
# parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--nb', type=int, default=4, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
# parser.add_argument('--train_epoch', type=int, default=100)
# parser.add_argument('--pre_train_epoch', type=int, default=10)
# parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
# parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
# parser.add_argument('--con_lambda', type=float, default=10, help='lambda for content loss')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
# parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--pre_trained_model', required=True, default='pre_trained_model', help='pre_trained cartoongan model path')
parser.add_argument('--image_dir', required=True, default='image_dir', help='test image path')
parser.add_argument('--output_image_dir', required=False, default='output_image_dir', help='output test image path')
parser.add_argument('--visual_image_dir', required=False, default='visual_image_dir', help='output test image path')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

G = networks.generator(args.in_ngc, args.out_ngc, args.ngf, args.nb)
if torch.cuda.is_available():
    G.load_state_dict(torch.load(args.pre_trained_model))
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

# if not os.path.exists(args.visual_image_dir):
#     os.mkdir(args.visual_image_dir)


for name in tqdm(name_list):
    load_path = os.path.join(args.image_dir, name)
    save_out_path = os.path.join(args.output_image_dir, name)
    # save_visual_path = os.path.join(args.visual_image_dir, name)

    raw_image = cv2.imread(load_path)[:,:,::-1]

    raw_image = Image.fromarray(raw_image)
    x = src_transform(raw_image).to(device)
    x = torch.unsqueeze(x, 0)
    G_recon = G(x)[0]
    plt.imsave(save_out_path[:-3] + 'png', (G_recon.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)




# # utils.data_load(os.path.join('data', args.src_data), 'test', src_transform, 1, shuffle=True, drop_last=True)
# image_src = utils.data_load(os.path.join(args.image_dir), 'test', src_transform, 1, shuffle=True, drop_last=True)

# with torch.no_grad():
#     G.eval()
#     for n, (x, _) in enumerate(image_src):
#         x = x.to(device)
#         print(type(x))
        # print(x.shape)
        # G_recon = G(x)
        # result = torch.cat((x[0], G_recon[0]), 2)
        # path = os.path.join(args.output_image_dir, str(n + 1) + '.png')
        # plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)


