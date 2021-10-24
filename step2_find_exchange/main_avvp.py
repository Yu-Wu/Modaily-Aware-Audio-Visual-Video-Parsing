from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import *
from nets.net_audiovisual import MMIL_Net
from utils.eval_metrics import segment_level, event_level
import pandas as pd
import pickle as pkl


def get_modality_aware_label(args, model, train_loader, optimizer, criterion, epoch):
    model.eval()
    v_accs = []
    a_accs = []
    num = 0

    das = []
    for batch_idx, sample in enumerate(train_loader):
        audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['video_st'].to('cuda'), sample['label'].type(torch.FloatTensor).to('cuda')
        audio2 = sample['audio2'].to('cuda')
        label2 = sample['label2']
        video2 = sample['video_s2'].to('cuda')
        video_st2 = sample['video_st2'].to('cuda')
        data_idx = sample['data_idx']
        optimizer.zero_grad()
        output, a_prob, v_prob, _, sims, mask = model(audio, video, video_st, audio2)
        output2, a_prob2, v_prob2, _, _, _ = model(audio, video2, video_st2, audio2)
        output3, a_prob3, v_prob3, _, _, _ = model(audio2, video, video_st, audio2)


        a_v = a_prob2
        v_v = v_prob2

        a_a = a_prob3
        v_a = v_prob3

        da = {
              'a': a_prob.cpu().detach(),
              'v': v_prob.cpu().detach(),
              'a_v':a_v.cpu().detach(),
              'v_v': v_v.cpu().detach(),
              'a_a': a_a.cpu().detach(),
              'v_a':v_a.cpu().detach(), 
              'label':target.cpu(), 'label2':label2, 'idx': data_idx}
        das.append(da)
        output.clamp_(min=1e-7, max=1 - 1e-7)
        a_prob.clamp_(min=1e-7, max=1 - 1e-7)
        v_prob.clamp_(min=1e-7, max=1 - 1e-7)

        data_idx = sample['data_idx']
        b=audio.size(0)

        acc = (torch.argmax(sims[0], dim=-1) == mask).float().mean().item()

        v_acc = (v_prob>0.5) == target
        a_acc = (a_prob>0.5) == target

        v_acc = v_acc.cpu().float()
        a_acc = a_acc.cpu().float()
        v_accs.append(v_acc)
        a_accs.append(a_acc)

        if batch_idx % args.log_interval == 0:
            print('Estimate Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader)))
    v_accs = torch.cat(v_accs, dim=0).mean(0)
    a_accs = torch.cat(a_accs, dim=0).mean(0)


    estimate_label(das, v_accs, a_accs)



def estimate_label(datas, v_accs, a_accs):

    v_err = 1 - torch.Tensor(v_accs)
    a_err = 1 - torch.Tensor(a_accs)

    v_class = v_err / torch.max(v_err)    
    a_class = a_err / torch.max(a_err)
   
    need_to_change_v = [[] for _ in range(25) ]
    need_to_change_a = [[] for _ in range(25) ]
    total_a = 0
    total_v = 0
    changed_a = 0
    changed_v = 0
    for data in datas:
        a = data['a']
        v = data['v']
        a_v = data['a_v']
        v_v = data['v_v']
        a_a = data['a_a']
        v_a = data['v_a']
    
        label = data['label']
        idx = data['idx']
    
    
        a = a * label
        v = v * label
        a_v = a_v * label
        v_v = v_v * label
        a_a = a_a * label
        v_a = v_a * label
    
        for b in range(len(a)):
          for c in range(25):
            if label[b][c] != 0:
                if v_a[b][c]/v_class[c] < 0.5:
                    if a_a[b][c]/ v_class[c] < 0.5:
                        # visual is not correct, given original visual data is input.
                        need_to_change_v[c].append(idx[b])

                if a_v[b][c]/a_class[c] < 0.5:
                    if v_v[b][c] /a_class[c] < 0.5:
                        need_to_change_a[c].append(idx[b])
       
    
    with open("need_to_change.pkl", 'wb') as f:
        pkl.dump([need_to_change_v, need_to_change_a], f)


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Video Parsing')
    parser.add_argument(
        "--audio_dir", type=str, default='data/feats/vggish/', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default='data/feats/res152/',
        help="video dir")
    parser.add_argument(
        "--st_dir", type=str, default='data/feats/r2plus1d_18/',
        help="video dir")
    parser.add_argument(
        "--label_train", type=str, default="data/AVVP_train.csv", help="weak train csv file")
    parser.add_argument(
        "--label_val", type=str, default="data/AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument(
        "--label_test", type=str, default="data/AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='MMIL_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='MMIL_Net',
        help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')
    args = parser.parse_args()

    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)

    if args.model == 'MMIL_Net':
        model = MMIL_Net().to('cuda')
    else:
        raise ('not recognized')

    if args.mode == 'estimate_labels':
        train_dataset = LLP_dataset(train=True, label=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([
                                               ToTensor()]))
        val_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([
                                               ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.BCELoss()
        best_F = 0
        test_dataset = LLP_dataset(train=False, label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,
            st_dir=args.st_dir, transform=transforms.Compose([
            ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        get_modality_aware_label(args, model, train_loader, optimizer, criterion, epoch=1)
            
if __name__ == '__main__':
    main()
