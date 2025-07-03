import argparse
import matplotlib.pyplot as plt
import itertools
import time
import datetime
import sys
import torchvision
from scipy.io import savemat
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import *
from tempdataset import *
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=220, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")  # 初始0.0002
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=80, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=10, help="number of image channels")
    parser.add_argument(
        "--sample_interval", type=int, default=2500, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    # criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_Argodot = torch.nn.MSELoss()
    criterion_pixelwise2 = torch.nn.MSELoss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 120
    # L1,L2 Regularization
    lambda_l1 = 0.001

    # 定义正交正则化
    def orthogonal_reg(W, device='cuda'):
        """计算权重矩阵W的正交正则化损失"""
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        WWT = torch.matmul(W, W.t())
        I = torch.eye(WWT.size(0), device=device)
        loss = torch.norm(WWT - I, p='fro')
        return loss

    # 正交正则化权重
    lambda_ortho = 0.001

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, int((opt.img_height // 2 ** 4)), int((opt.img_width // 2 ** 4)))

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load(
            r'C:\Users\admin\PycharmProjects\pythonProject\saved_models/%s/generator_%d.pth' % (
                opt.dataset_name, opt.epoch)))
        discriminator.load_state_dict(torch.load(
            r'C:\Users\admin\PycharmProjects\pythonProject\saved_models/%s/discriminator_%d.pth' % (
                opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=0.005)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=0.05)
    # Learning rate schedulers
    scheduler_G = StepLR(optimizer_G, step_size=20, gamma=0.5)  # 每50个epoch学习率乘以0.1
    scheduler_D = StepLR(optimizer_D, step_size=20, gamma=0.5)
    dataloader = DataLoader(
        ImageDataset(r'C:\GANdata', transforms_=None),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    val_dataloader = DataLoader(
        ImageDataset(r'C:\GANdata', transforms_=None, mode="val"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=16,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    for epoch in range(opt.epoch, opt.n_epochs):
        if epoch > 30:
            Lambda_Argo = 0.8
            lambda_pixel1 = lambda_pixel * 1
        else:
            Lambda_Argo = 0.2
            lambda_pixel1 = lambda_pixel * 1
        for i, batch in enumerate(dataloader):
            # Model inputs
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            Argo_data = Variable(batch["Argo"].type(Tensor))
            season_label = Variable(batch["season_label"].type(Tensor))
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A, season_label)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            loss_pixel2 = criterion_pixelwise2(fake_B, real_B)
            pixel_total = loss_pixel * 0.6 + loss_pixel2 * 0.4

            # Total loss
            loss_G = loss_GAN + lambda_pixel1 * pixel_total
            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # 使用正交正则化
            loss_orthoD = torch.tensor(0.).cuda()
            for param in discriminator.parameters():
                if param.requires_grad and param.dim() > 1:
                    loss_orthoD += orthogonal_reg(param, device='cuda' if cuda else 'cpu')

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake) + lambda_ortho * loss_orthoD

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] "
                "[D loss: %f, loss_real: %f, loss_fake: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    loss_D.item(),
                    loss_real.item(),
                    loss_fake.item(),
                    time_left,
                )
            )
        scheduler_G.step()
        scheduler_D.step()
        # Optional: Log the current learning rate
        print("Epoch:", epoch, "LR_G:", scheduler_G.get_last_lr(), "LR_D:", scheduler_D.get_last_lr())
        with torch.no_grad():
            val_mse = 0.0
            val_dotloss = 0
            val_samples = 0
            for val_batch in val_dataloader:
                real_A_val = Variable(val_batch["A"].type(Tensor))
                real_B_val = Variable(val_batch["B"].type(Tensor))
                Argo_data1 = Variable(val_batch["Argo"].type(Tensor))
                season_label_val = Variable(val_batch["season_label"].type(Tensor))
                fake_B_val = generator(real_A_val, season_label_val)
                # 创建掩码
                # mask = real_B_val != 0
                # fake_B_val = fake_B_val[mask]
                # real_B_val = real_B_val[mask]
                mse = criterion_pixelwise2(fake_B_val, real_B_val).item()  # 计算MSE
                val_mse += mse
                val_samples += 1
                # Argo dot loss
                mask = Argo_data1 != 0
                # 过滤有效的输出和真实值
                filtered_Argo = Argo_data1[mask]
                fake_B_val = fake_B_val[mask]
                dot_loss = criterion_Argodot(fake_B_val, filtered_Argo)
                val_dotloss += dot_loss
            val_mse /= val_samples
            val_dotloss /= val_samples
            print(f"Epoch {epoch}: Validation MSE = {val_mse},dot_loss = {dot_loss}")
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))


if __name__ == '__main__':
    main()
