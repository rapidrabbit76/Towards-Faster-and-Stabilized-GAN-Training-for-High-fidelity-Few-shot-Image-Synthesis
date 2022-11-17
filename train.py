from asyncore import write
import os
import argparse
import random

import hydra
import torch
import torch.nn.functional as F
import torch.optim as optim
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm import tqdm

from diffaug import DiffAugment
from hyperparameters import Hyperparameters
from models import Discriminator, Generator, weights_init
from operation import (
    seed_everything,
    ImageFolder,
    InfiniteSamplerWrapper,
    copy_G_params,
    get_dir,
    load_params,
)
from torch.utils.tensorboard import SummaryWriter


policy = "color,translation"
import lpips


percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)


def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = (
            F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean()
            + percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum()
            + percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum()
            + percept(
                rec_part,
                F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]),
            ).sum()
        )
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()


@hydra.main(
    config_path="./",
    config_name="config.yaml",
)
def main(hp: Hyperparameters) -> None:
    seed_everything(hp.seed)
    writer = SummaryWriter()
    device = torch.device(hp.device)

    transform_list = [
        transforms.Resize((int(hp.image_size), int(hp.image_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    trans = transforms.Compose(transform_list)
    dataset = ImageFolder(root=hp.data_root, transform=trans)

    dataloader = iter(
        DataLoader(
            dataset,
            batch_size=hp.batch_size,
            shuffle=False,
            sampler=InfiniteSamplerWrapper(dataset),
            num_workers=hp.num_workers,
            pin_memory=True,
        )
    )
    # from model_s import Generator, Discriminator
    netG = Generator(ngf=hp.ngf, nz=hp.nz, im_size=hp.image_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=hp.ndf, im_size=hp.image_size)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, hp.nz).normal_(0, 1).to(device)

    optimizerG = optim.Adam(
        netG.parameters(), lr=hp.lr, betas=(hp.beta.beta_1, hp.beta.beta_2)
    )
    optimizerD = optim.Adam(
        netD.parameters(), lr=hp.lr, betas=(hp.beta.beta_1, hp.beta.beta_2)
    )
    current_iteration = hp.start_iter
    if hp.checkpoint:
        ckpt = torch.load(hp.checkpoint, map_location=device)
        netG.load_state_dict(
            {k.replace("module.", ""): v for k, v in ckpt["g"].items()}
        )
        netD.load_state_dict(
            {k.replace("module.", ""): v for k, v in ckpt["d"].items()}
        )
        avg_param_G = ckpt["g_ema"]
        optimizerG.load_state_dict(ckpt["opt_g"])
        optimizerD.load_state_dict(ckpt["opt_d"])
        current_iteration = int(hp.checkpoint.split("_")[-1].split(".")[0])
        del ckpt

    netG = nn.DataParallel(netG.to(device))
    netD = nn.DataParallel(netD.to(device))

    pbar = tqdm(range(current_iteration, hp.total_iterations + 1))
    for iteration in pbar:
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, hp.nz).normal_(0, 1).to(device)

        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(
            netD, real_image, label="real"
        )
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()

        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        pbar.set_description_str(
            f"G_LOSS : {err_dr: .5f} , D_LOSS: {-err_g.item(): .5f}"
        )

        # EMA
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            writer.add_scalar("Loss/G", -err_g.item(), iteration)
            writer.add_scalar("Loss/D", err_dr, iteration)

        if iteration % hp.interval.sample == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.inference_mode():
                pred_image = netG(fixed_noise)[0].add(1).mul(0.5)
                pred_image = vutils.make_grid(
                    pred_image,
                    nrow=4,
                    padding=1,
                )
                writer.add_image("sample", pred_image, iteration)
            load_params(netG, backup_para)

        if iteration % hp.interval.save == 0 or iteration == hp.total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save(
                {"g": netG.state_dict(), "d": netD.state_dict()},
                "%d.pth" % iteration,
            )
            load_params(netG, backup_para)
            torch.save(
                {
                    "g": netG.state_dict(),
                    "d": netD.state_dict(),
                    "g_ema": avg_param_G,
                    "opt_g": optimizerG.state_dict(),
                    "opt_d": optimizerD.state_dict(),
                },
                "all_%d.pth" % iteration,
            )


if __name__ == "__main__":
    main()
