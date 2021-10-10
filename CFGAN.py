import os, time, pickle, argparse, networks, utils
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from edge_promoting import edge_promoting
import torch.nn.functional as F
import os 


parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='project_name',  help='')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--con_lambda', type=float, default=100, help='lambda for content loss')
parser.add_argument('--tv_lambda', type=float, default=0, help='lambda for tv loss')
parser.add_argument('--adv_lambda', type=float, default=2, help='lambda for advD loss')

parser.add_argument('--data_path', required=True, default='data/selfie2anime',  help='data path')

parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')

parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--nb', type=int, default=4, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')

parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--pre_train_epoch', type=int, default=30)

parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

parser.add_argument('--latest_generator_model', required=False, default='', help='the latest trained model path')
parser.add_argument('--latest_discriminator_model', required=False, default='', help='the latest trained model path')
args = parser.parse_args()




print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

print(device)


def total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


# results save path
if not os.path.isdir(os.path.join(args.name + '_results', 'Reconstruction')):
    os.makedirs(os.path.join(args.name + '_results', 'Reconstruction'))
if not os.path.isdir(os.path.join(args.name + '_results', 'Transfer')):
    os.makedirs(os.path.join(args.name + '_results', 'Transfer'))


# data_loader
transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainA = utils.data_load(os.path.join('data', args.data_path), 'testA', transform, args.batch_size, shuffle=True, drop_last=True)
trainB = utils.data_load(os.path.join('data', args.data_path), 'trainB', transform, args.batch_size, shuffle=True, drop_last=True)
testA = utils.data_load(os.path.join('data', args.data_path), 'testA', transform, 1, shuffle=True, drop_last=True)


# network
G = networks.generator(args.in_ngc, args.out_ngc, args.ngf, args.nb)
if args.latest_generator_model != '':
    if torch.cuda.is_available():
        G.load_state_dict(torch.load(args.latest_generator_model))
    else:
        G.load_state_dict(torch.load(args.latest_generator_model, map_location=lambda storage, loc: storage))

D = networks.discriminator(args.in_ndc, args.out_ndc, args.ndf)
if args.latest_discriminator_model != '':
    if torch.cuda.is_available():
        D.load_state_dict(torch.load(args.latest_discriminator_model))
    else:
        D.load_state_dict(torch.load(args.latest_discriminator_model, map_location=lambda storage, loc: storage))

RestNet18 = networks.RestNet18(init_weights=None)

G.to(device)
D.to(device)
RestNet18.to(device)
G.train()
D.train()
RestNet18.eval()

print('---------- Networks initialized -------------')
utils.print_network(G)
utils.print_network(D)
utils.print_network(RestNet18)
print('-----------------------------------------------')

# loss
BCE_loss = nn.BCELoss().to(device)
L1_loss = nn.L1Loss().to(device)

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
D_optimizer = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)

pre_train_hist = {}
pre_train_hist['Recon_loss'] = []
pre_train_hist['per_epoch_time'] = []
pre_train_hist['total_time'] = []

""" Pre-train reconstruction """
if args.latest_generator_model == '':
    print('Pre-training start!')
    start_time = time.time()
    for epoch in range(args.pre_train_epoch):
        epoch_start_time = time.time()
        Recon_losses = []
        for x, _ in trainA:
            x = x.to(device)
            
            G_optimizer.zero_grad()

            G_ = G(x)
            Recon_loss = args.con_lambda * L1_loss(G_, x)

            Recon_losses.append(Recon_loss.item())
            pre_train_hist['Recon_loss'].append(Recon_loss.item())

            Recon_loss.backward()
            G_optimizer.step()

        per_epoch_time = time.time() - epoch_start_time
        pre_train_hist['per_epoch_time'].append(per_epoch_time)
        print('[%d/%d] - time: %.2f, Recon loss: %.3f' % ((epoch + 1), args.pre_train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Recon_losses))))

    total_time = time.time() - start_time
    pre_train_hist['total_time'].append(total_time)
    with open(os.path.join(args.name + '_results',  'pre_train_hist.pkl'), 'wb') as f:
        pickle.dump(pre_train_hist, f)

    with torch.no_grad():
        G.eval()
        torch.save(G.state_dict(), os.path.join(args.name + '_results', 'pretrained_generator.pkl'))
        for n, (x, _) in enumerate(trainA):
            x = x.to(device)
            G_recon = G(x)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(args.name + '_results', 'Reconstruction', args.name + '_train_recon_' + str(n + 1) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            if n == 4:
                break

        for n, (x, _) in enumerate(testA):
            x = x.to(device)
            G_recon = G(x)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(args.name + '_results', 'Reconstruction', args.name + '_test_recon_' + str(n + 1) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            if n == 4:
                break
else:
    print('Load the latest generator model, no need to pre-train')



# train_hist = {}
# train_hist['Disc_loss'] = []
# train_hist['Gen_loss'] = []
# train_hist['Con_loss'] = []
# train_hist['per_epoch_time'] = []
# train_hist['total_time'] = []
# print('training start!')
# start_time = time.time()
# real = torch.ones(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
# fake = torch.zeros(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
# for epoch in range(args.train_epoch):
#     epoch_start_time = time.time()
#     G.train()
#     Disc_losses = []
#     Gen_losses = []
#     Con_losses = []
#     for (x, _), (y, _) in zip(pretrain_loader_src, train_loader_tgt):
#         x_sm = x[:, :, :, args.input_size:]
#         x = x[:, :, :, :args.input_size]
#         # e = y[:, :, :, args.input_size:]
#         y = y[:, :, :, :args.input_size]
#         x, y, x_sm = x.to(device), y.to(device), x_sm.to(device)

#         # train D
#         D_optimizer.zero_grad()

#         D_real = D(y)
#         D_real_loss = BCE_loss(D_real, real)

#         G_ = G(x)
#         D_fake = D(G_)
#         D_fake_loss = BCE_loss(D_fake, fake)
#         Disc_loss = (D_real_loss + D_fake_loss)*args.adv_lambda

#         Disc_losses.append(Disc_loss.item())
#         train_hist['Disc_loss'].append(Disc_loss.item())

#         Disc_loss.backward()
#         D_optimizer.step()

#         # train G
#         G_optimizer.zero_grad()

#         G_ = G(x)
#         D_fake = D(G_)
#         D_fake_loss = BCE_loss(D_fake, real)*args.adv_lambda


#         x_feature = RestNet18((x_sm + 1) / 2)
#         G_feature = RestNet18((G_ + 1) / 2)
#         Con_loss = args.con_lambda * L1_loss(G_feature, x_feature.detach())


#         tv_loss = total_variation_loss(G_, args.tv_lambda)

#         Gen_loss = D_fake_loss + Con_loss + tv_loss
#         Gen_losses.append(D_fake_loss.item())
#         train_hist['Gen_loss'].append(D_fake_loss.item())
#         Con_losses.append(Con_loss.item())
#         train_hist['Con_loss'].append(Con_loss.item())

#         Gen_loss.backward()
#         G_optimizer.step()

#     G_scheduler.step()
#     D_scheduler.step()


#     per_epoch_time = time.time() - epoch_start_time
#     train_hist['per_epoch_time'].append(per_epoch_time)
#     print(
#     '[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f' % ((epoch + 1), args.train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Disc_losses)),
#         torch.mean(torch.FloatTensor(Gen_losses)), torch.mean(torch.FloatTensor(Con_losses))))

#     if epoch % 2 == 1 or epoch == args.train_epoch - 1:
#         with torch.no_grad():
#             G.eval()
#             for n, (x, _) in enumerate(pretrain_loader_src):
#                 x_sm = x[:, :, :, args.input_size:]
#                 x = x[:, :, :, :args.input_size]
#                 x, x_sm = x.to(device), x_sm.to(device)

#                 G_recon = G(x)
#                 result = torch.cat((x[0], G_recon[0]), 2)
#                 path = os.path.join(args.name + '_results', 'Transfer', str(epoch+1) + '_epoch_' + args.name + '_train_' + str(n + 1) + '.png')
#                 plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
#                 if n == 4:
#                     break

#             for n, (x, _) in enumerate(test_loader_src):
#                 x = x.to(device)
#                 G_recon = G(x)
#                 result = torch.cat((x[0], G_recon[0]), 2)
#                 path = os.path.join(args.name + '_results', 'Transfer', str(epoch+1) + '_epoch_' + args.name + '_test_' + str(n + 1) + '.png')
#                 plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
#                 if n == 4:
#                     break

#             torch.save(G.state_dict(), os.path.join(args.name + '_results', 'generator_latest.pkl'))
#             torch.save(D.state_dict(), os.path.join(args.name + '_results', 'discriminator_latest.pkl'))

# total_time = time.time() - start_time
# train_hist['total_time'].append(total_time)

# print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), args.train_epoch, total_time))
# print("Training finish!... save training results")

# torch.save(G.state_dict(), os.path.join(args.name + '_results',  'generator_param.pkl'))
# torch.save(D.state_dict(), os.path.join(args.name + '_results',  'discriminator_param.pkl'))
# with open(os.path.join(args.name + '_results',  'train_hist.pkl'), 'wb') as f:
#     pickle.dump(train_hist, f)
