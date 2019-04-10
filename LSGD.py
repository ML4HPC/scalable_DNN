import argparse
import os
import math
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrpower', '--learning-rate-power', default=1, type=int,
                    metavar='LRPOWER', help='power of poly learning rate policy')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
#parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--gpu-num', default=None, type=int, help='The number of GPUs in a node.')
parser.add_argument('--cuda', type=str, default=True)

best_prec1 = 0

rank = 0
world_size = 0
node_num = 0
local_size = 0
local_rank = 0
train_proc_num = 0
node_idx = 0
comm_rank = False
local_root = 0



def avg_grad(model, node_handle):

    #print('avg_grad() : train_proc_num : ', train_proc_num)

    #dist.barrier();
    for param in model.parameters():
        #print('RANK[',rank,']: param.nelement() : ', param.nelement(), ', param.grad.data type : ', type(param.grad.data), ', param.grad.data size : ' , param.grad.data.size())
        #dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=node_handle)
        dist.reduce(param.grad.data, local_root, op=dist.ReduceOp.SUM, group=node_handle)
        #print('RANK[',rank,']: avg_grad()? - 2')
        #param.grad.data /= float(train_proc_num)



def avg_grad_comm(model, node_handle):

    #print('avg_grad_comm() : train_proc_num : ', train_proc_num)
    #dist.barrier();
    for param in model.parameters():
        #dist.all_reduce(param.data, op=dist.reduce_op.SUM, group=node_handle)
        dist.reduce(param.data, local_root, op=dist.ReduceOp.SUM, group=node_handle)
        param.data /= float(train_proc_num)


def reduce_loss(total_loss, n_samples):
    reduction = torch.FloatTensor([total_loss,n_samples])
    dist.all_reduce(reduction, op=dist.ReduceOp.SUM)
    if rank==0: print('n_samples : ', int(reduction[1].item()))
    return float(reduction[0].item() / reduction[1].item())



def main():
    global args, best_prec1
    args = parser.parse_args()

    global rank, world_size, node_num, local_size, local_rank, train_proc_num, node_idx, local_root

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_size = (int)(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    local_rank = (int)(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    node_num = world_size // local_size

    train_proc_num = world_size - node_num
    node_idx = rank // local_size

    local_root = node_idx * local_size

    #test = torch.FloatTensor([1234567, 7654321])
    #print('test[0] : ', test[0])
    #print('test[0].item() : ', test[0].item())

    start_idx = 0
    end_idx = local_size
    #node_handle = list(node_num)
    node_handle = list()
    comm_handle_idx = list()
    for i in range(node_num):
        node_handle.append(dist.new_group(range(start_idx, end_idx)))
        comm_handle_idx.append(start_idx)
        start_idx += local_size
        end_idx += local_size

    comm_handle = dist.new_group(comm_handle_idx)


    if rank==0: print('comm_handle : ', comm_handle_idx)
    if rank==0: print('pytorch version : ', torch.__version__)
    if rank==0: print('cuDNN version : ', torch.backends.cudnn.version())
    if rank==0: print('WORLD SIZE:', world_size)
    if rank==0: print('The number of nodes : ', node_num)
    if rank==0: print('The number of ranks in a node : ', local_size)
    if rank==0: print('The number of processes doing training : ', train_proc_num)




    args.cuda = args.gpu_num is not None
    if args.cuda:
        if local_rank > 0 and local_rank <= args.gpu_num:
            torch.cuda.set_device(local_rank-1)
        else:
            args.cuda=False

    comm_rank = False
    if rank % local_size == 0:
        comm_rank = True



    dist.barrier();
    if rank==0: print('========================================================')
    dist.barrier();
    print('[',rank,'] : OMPI_COMM_WORLD_LOCAL_RANK : ', os.environ['OMPI_COMM_WORLD_LOCAL_RANK'], ', Node : ', node_idx, ', GPU : ', args.cuda, ', COMM_RANK : ', comm_rank)
    dist.barrier();
    if rank==0: print('========================================================')
    dist.barrier();







    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    #if args.gpu is not None:
    #    warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    #args.distributed = args.world_size > 1

    #if args.distributed:
    #    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                            world_size=args.world_size)

    # create model
    if args.pretrained:
        if rank==0: print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        if rank==0: print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    #if args.gpu is not None:
    #    model = model.cuda(args.gpu)
    #elif args.distributed:
    #    model.cuda()
    #    model = torch.nn.parallel.DistributedDataParallel(model)
    #else:
    #    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #        model.features = torch.nn.DataParallel(model.features)
    #        model.cuda()
    #    else:
    #        model = torch.nn.DataParallel(model).cuda()
    if args.cuda:
        model.cuda()




    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    #if args.cuda:
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if comm_rank == False:
            if os.path.isfile(args.resume):
                if rank==1: print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if rank==1: print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    if comm_rank:
        for param in model.parameters():
            param.data.zero_()

    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        if comm_rank:
            param.data.zero_()
        else:
            param.data /= train_proc_num


    cudnn.benchmark = True



    # Data loading code
  
    train_loader = None

    traindir = os.path.join(args.data, 'ILSVRC2012_img_train')
    valdir = os.path.join(args.data, 'ILSVRC2012_img_val')
    if rank==0: print('train image path : ', str(traindir))
    if rank==0: print('validation image path : ', str(valdir))

    if comm_rank == False:

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        #if args.distributed:
        #    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #else:
        #    train_sampler = None
        #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, train_proc_num, rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, train_proc_num, (args.gpu_num*node_idx + local_rank - 1))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=0, pin_memory=True, sampler=train_sampler)


        if rank==1 : 
            print('train_loader length : ', len(train_loader))
            print('train_loader length : ', math.ceil(1281168/float(args.batch_size * train_proc_num)))
            print('(node_num * args.gpu_num) : ', (node_num * args.gpu_num))
            print('train_proc_num : ', train_proc_num)

            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=True)


    #if args.evaluate:
    #    validate(val_loader, model, criterion)
    #    return

    dist.barrier()




    total_start = time.time();
    total_training = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        if comm_rank == False: train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lrpower)

        # train for one epoch
        if rank==0: print('train of epoch ' + str(epoch + 1) + ' start!!!')
        start = time.time()
        train_loss = train(comm_rank, node_handle, node_idx, comm_handle, train_loader, model, criterion, optimizer, epoch)
        end = time.time()
        total_training = total_training + (end - start)
        if rank==0: print('Epoch ['+str(epoch+1)+'] time : ' + str(end - start) + ' with loss : ' + str(train_loss))

        if rank==1:
            # evaluate on validation set
            start = time.time()
            prec1 = validate(val_loader, model, criterion)
            print('Test Epoch ['+str(epoch+1)+'] time : ' + str(time.time() - start) + ' with prec1 : ' + str(prec1))

            ## remember best prec@1 and save checkpoint
            #is_best = prec1 > best_prec1
            #best_prec1 = max(prec1, best_prec1)
            #save_checkpoint({
            #    'epoch': epoch + 1,
            #    'arch': args.arch,
            #   'state_dict': model.state_dict(),
            #    'best_prec1': best_prec1,
            #    'optimizer' : optimizer.state_dict(),
            #}, is_best)

        dist.barrier()

    if rank==0: 
        print('Total training time : ', total_training)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
           'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, False)

def train(comm_rank, node_handle, node_idx, comm_handle, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    #train_loader_length = len(train_loader)
    train_loader_length = math.ceil(1281168/float(args.batch_size * train_proc_num))

    total_loss = 0;
    n_samples = 0;

    #print('train start!')


    if comm_rank:

        #start = time.time()
        for i in range(train_loader_length):

            #for param in model.parameters():
            #    param.data.zero_()

            #avg_grad_comm(model, node_handle[node_idx])
            for param in model.parameters():
                param.data.zero_()
                dist.reduce(param.data, local_root, op=dist.ReduceOp.SUM, group=node_handle[node_idx])
                param.data /= float(train_proc_num)
            #print('[',rank,'] here? - 3')

            for param in model.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=comm_handle)


            for param in model.parameters():
                dist.broadcast(param.data, local_root, group=node_handle[node_idx])


    else:

        # switch to train mode
        model.train()

        #print('rank:'+str(rank)+', comm_rank: '+str(comm_rank)+' train_loader type : ', type(train_loader))

        #end = time.time()
        i = 0
        for i, (input, target) in enumerate(train_loader):
            if epoch < 5:
                warmup_learning_rate(optimizer, train_loader_length, epoch, i)

            #print('[',rank,'] : input.size(0): ', input.size(0)
            # measure data loading time
            #data_time.update(time.time() - end)

            input = input.cuda()
            target = target.cuda()
 
            if i > 0:
                for param in model.parameters():
                    dist.broadcast(param.grad.data, local_root, group=node_handle[node_idx])
            
                optimizer.step()
                n_samples += output.size(0)
                total_loss += (loss.item() * output.size(0))
                #if rank==1: print('[i:',(i-1),'] losses.avg : ', losses.avg)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            #prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            #top1.update(prec1[0], input.size(0))
            #top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            #avg_grad(model, node_handle[node_idx]);
            for param in model.parameters():
                dist.reduce(param.grad.data, local_root, op=dist.ReduceOp.SUM, group=node_handle[node_idx])


            # measure elapsed time
            #batch_time.update(time.time() - end)
            #end = time.time()


        for param in model.parameters():
            dist.broadcast(param.grad.data, local_root, group=node_handle[node_idx])
            
        optimizer.step()
        n_samples += output.size(0)
        total_loss += (loss.item() * output.size(0))
        #if rank==1: print('[i:',(i-1),'] losses.avg : ', losses.avg)

        if rank==1: 
            print('losses.avg : ', losses.avg)
            print('total_loss / n_samples : ', float(total_loss / n_samples))


    return reduce_loss(total_loss, n_samples)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if i % args.print_freq == 0:
            #    if rank==0: print('Test: [{0}/{1}]\t'
            #          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            #           i, len(val_loader), batch_time=batch_time, loss=losses,
            #           top1=top1, top5=top5))

        if rank==1: print('Loss {loss.val:.4f} ({loss.avg:.4f}) \t Prec@1 {top1.avg:.3f} \t  Prec@5 {top5.avg:.3f}'.format(loss=losses, top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def warmup_learning_rate(optimizer, loader_len, epoch, it):
    base_lr = 0.05
    end_ep = 5

    if epoch < end_ep and args.lr > base_lr :
        total_grid = loader_len*end_ep
        lr = base_lr + ((it + loader_len*epoch)/float(total_grid))*(args.lr-base_lr)
        
        #print('warmup_learning_rate() : i='+str(it)+', lr=', lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



def adjust_learning_rate(optimizer, epoch, power):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (power*(epoch // 30)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

  
    dist.init_process_group('mpi')

    main()
