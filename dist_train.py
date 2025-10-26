import argparse
import os
import pdb
import warnings
import model.krccsnet as krccsnet
from loss import *
import torch.optim as optim
from data_processor import *
from trainer import *
import torch.nn.functional as F
from dist_trainer import *
warnings.filterwarnings("ignore")
from accelerate import Accelerator
import time
def main():
    global args
    args = parser.parse_args()
    # setup_seed(1)

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        torch.backends.cudnn.benchmark = True

    if args.model =='krccsnet_train':
        model = krccsnet.build_LKSN_ARM(sensing_rate=args.sensing_rate)

    criterion = loss_fn
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 90, 120, 150, 180], gamma=0.25, last_epoch=-1)
    train_loader, test_loader_bsds, test_loader_set5, test_loader_set14 = data_loader(args)
    accelerator=Accelerator(
            split_batches = True,
            # mixed_precision = 'fp16' if fp16 else 'no'
        )
    trainer=Trainer(
        model,
        criterion, 
        optimizer,
        train_loader, 
        test_loader_bsds, 
        test_loader_set5, 
        test_loader_set14,
        accelerator

    )
    accelerator.print('\n')
    accelerator.print(time.ctime(time.time()))
    accelerator.print('Model: %s\n'
          'Sensing Rate: %.6f\n'
          'Epoch: %d\n'
          'Initial LR: %f\n'
          % (args.model, args.sensing_rate, args.epochs, args.lr))

    accelerator.print('Start training at ')
    for epoch in range(args.epochs):
        accelerator.print('\ncurrent lr {:.5e}'.format(trainer.optimizer.param_groups[0]['lr']))
        loss = trainer.train_epoch(epoch)
        scheduler.step()
        trainer.valid(trainer.test_loader_bsds,'BSDS')
        trainer.valid(trainer.test_loader_set5,'Set5')
        trainer.valid(trainer.test_loader_set14,'Set14')
    trainer.save('./saved_model/' + str(args.model) + '_' + str(args.sensing_rate) + '.pth')
    accelerator.print('Trained finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='krccsnet_train',
                        choices=['krccsnet_train'],
                        help='choose model to train')
    parser.add_argument('--sensing-rate', type=float, default=0.5,
                        choices=[0.50000, 0.25000, 0.12500, 0.06250, 0.03125, 0.015625],
                        help='set sensing rate')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--block-size', default=32, type=int,
                        metavar='N', help='block size (default: 32)')
    parser.add_argument('--image-size', default=96, type=int,
                        metavar='N', help='image size used for training (default: 96)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained model',
                        default='save_temp', type=str)

    main()
