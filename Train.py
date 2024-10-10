import argparse
from torch.utils.data import DataLoader
from datasets.potsdam import *
from datasets.loveda import *
from datasets.openearthmap import *
from datasets.earthmap import *
from thop import profile, clever_format
from model.UNetFormer import UNetFormer
from All_a_UNet import all_attention_UNet
from remotenet.RemoteNet import RemoteNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from tqdm import tqdm
import os
import torch
from timm.optim.lookahead import Lookahead
from timm.optim.lion import Lion
import torch.multiprocessing
from termcolor import colored 

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.enabled = False


def create_model(num_classes):
    model = UNetFormer(
        decode_channels=128,
        dropout=0.1,
        backbone_name='swsl_resnet50',
        pretrained=True,
        window_size=8,
        num_classes=num_classes)
    return model


def select_optimizer(optimizer_name, params_to_optimize, lr, weight_decay):
    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RAdam':
        return torch.optim.RAdam(params_to_optimize, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Lion':
        return Lion(params_to_optimize, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def main(args):
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    epochs = args.epochs

    # 加载数据集
    if args.dataset == "Loveda":
        train_dataset = LoveDATrainDataset(data_root=args.train_data_root)
        val_dataset = LoveDATrainDataset(data_root=args.val_data_root)
    elif args.dataset == "Potsdam":
        train_dataset = PotsdamTrainDataset(data_root=args.train_data_root)
        val_dataset = PotsdamValDataset(data_root=args.val_data_root)
    elif args.dataset == "openearthmap":
        train_dataset = PotsdamTrainDataset(data_root=args.train_data_root)
        val_dataset = PotsdamValDataset(data_root=args.val_data_root)
    elif args.dataset == "earthvqa":
        train_dataset = VQADataset(data_root=args.train_data_root)
        val_dataset = PotsdamValDataset(data_root=args.val_data_root)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    num_workers = args.num_workers

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=args.num_classes)

    model.to(device)

    checkpoint = torch.load(args.checkpoint)
    new_model = {f"decoder.MAE.MAE.{key}": value for key, value in checkpoint.items()}
    checkpoint['model'] = new_model

    model.load_state_dict(checkpoint, strict=False)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # 选择优化器
    optimizer = select_optimizer(args.optimizer, params_to_optimize, lr=args.lr, weight_decay=0.01)
    optimizer = Lookahead(optimizer)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_miou = 0

    for epoch in range(0, epochs):
        print(colored(f"\nEpoch [{epoch + 1}/{epochs}]", "cyan"))
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        mean_loss, lr, con = train_one_epoch(model, optimizer, train_loader, device,
                                             epoch, lr_scheduler=lr_scheduler,
                                             scaler=scaler, num_classes=args.num_classes,
                                             progress_bar=progress_bar)
        
        print(f"Epoch [{epoch + 1}/{epochs}] | Mean Loss: {mean_loss:.4f} | Learning Rate: {lr:.6f}")

        # 验证阶段
        print(colored("Validating...", "yellow"))
        confmat = evaluate(model, val_loader, device=device, num_classes=args.num_classes)
        val_info = str(confmat)

        # 获取当前mIoU
        miou = float(val_info.split('\n')[-5].split(': ')[-1])
        print(f"Validation mIoU: {miou:.4f}")

        # 保存最优模型
        if miou > best_miou:
            best_miou = miou
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(save_file, args.best_model_save_path)
            print(colored(f"New best mIoU: {best_miou:.4f}. Model saved to {args.best_model_save_path}.", "green"))

        if epoch == epochs - 1:
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(save_file, args.last_model_save_path)
            print(colored(f"Final model saved to {args.last_model_save_path}.", "green"))

        torch.cuda.empty_cache()

    print(colored(f"\nTraining complete. Best mIoU: {best_miou:.4f}", "magenta"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a UNetFormer model.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name, e.g., Loveda or Potsdam')
    parser.add_argument('--train_data_root', type=str, required=True, help='Path to the training data')
    parser.add_argument('--val_data_root', type=str, required=True, help='Path to the validation data')
    parser.add_argument('--train_batch_size', type=int, default=14, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=2, help='Validation batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of output classes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--best_model_save_path', type=str, default='lovedabest_model.pth', help='Path to save the best model')
    parser.add_argument('--last_model_save_path', type=str, default='lovedalast_model.pth', help='Path to save the last model')

    # 新增优化器选择
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'RAdam', 'Lion'], help='Optimizer to use')

    args = parser.parse_args()
    main(args)
