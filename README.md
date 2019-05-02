# scalable_DNN
Scalable DNN

Both LSGD.py and CSGD.py are variation based on https://github.com/pytorch/examples/blob/master/imagenet/main.py.
Hence, both needs torchvison to run.

CSGD.py is a conventional synchronous distributed SGD implementation for baseline test.
LSGD.py is the Layered SGD implmentation.

Usage:
When one node has four GPUs and 64 nodes (256 worker) are used in the running:

mpirun -np 320 python LSGD.py -a resnet50 --epoch 100 --batch-size 64 --gpu-num 4 --lr 6.4 [ImageNet path]
mpirun -np 256 python CSGD.py -a resnet50 --epoch 100 --batch-size 64 --lr 6.4 [ImageNet path]


Since LSGD need one additional CPU communicator, it needs one additional MPI process per node.
Since the number of node is 64, it should be 256 + 64 => 320.
When you use 512 node having 8 GPUs, then the number of MPI process you need is 9*512 => 4608.

