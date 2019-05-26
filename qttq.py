import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import torch.nn.functional as F

#----------导入resnet模型-------------
from net.Resnet import Resnet18PlusLatent
from model import *
bits=12


def get_model(learning_rate=0.00001):
    model=DSH(bits)
    model.load_state_dict(torch.load("./models/qttq/{}/bits{}.pkl".format(bits,bits)))

    #parmas don't need quantization
    other=[p for n,p in model.named_parameters() if "fc.2" not in n]

    #params of conv layer except the first conv layer
    weights_to_quantized=[p for n,p in model.named_parameters() if 'fc.2' in n]

    #all params
    #
    params=[{'params':other},
    {'params':weights_to_quantized}]

    optimizer=torch.optim.Adam(params,lr=learning_rate)

    loss=torch.nn.MSELoss().cuda()
    model.cuda()

    return model,loss,optimizer

#-----------------------------------


#----------相关函数、类定义------------
def initial_scales(kernel):
    return 1.0, 1.0


def quantize(kernel, w_p, w_n, t):
    """
    Return quantized weights of a layer.
    Only possible values of quantized weights are: {zero, w_p, -w_n}.
    """
    delta = t*kernel.abs().max()
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    return (w_p*a.cpu()).cuda() + (-w_n*b.cpu()).cuda()


def get_grads(kernel_grad, kernel, w_p, w_n, t):
    """
    Arguments:
        kernel_grad: gradient with respect to quantized kernel.
        kernel: corresponding full precision kernel.
        w_p, w_n: scaling factors.
        t: hyperparameter for quantization.

    Returns:
        1. gradient for the full precision kernel.
        2. gradient for w_p.
        3. gradient for w_n.
    """
    delta = t*kernel.abs().max()
    # masks
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    c = torch.ones(kernel.size()).cuda() - a - b
    # scaled kernel grad and grads for scaling factors (w_p, w_n)
    return w_p.cuda()*a*kernel_grad + w_n.cuda()*b*kernel_grad + 1.0*c*kernel_grad,\
        (a*kernel_grad).sum(), (b*kernel_grad).sum()


def optimization_step(model, loss, x_batch, y_batch, optimizer_list, t):
    """Make forward pass and update model parameters with gradients."""

    # parameter 't' is a hyperparameter for quantization

    # 'optimizer_list' contains optimizers for
    # 1. full model (all weights including quantized weights),
    # 2. backup of full precision weights,
    # 3. scaling factors for each layer
    optimizer, optimizer_fp, optimizer_sf = optimizer_list

    x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))
    # forward pass using quantized model
    _,tout,_,_=teacher(x_batch)
    tout=tout.detach()
    logits= model(x_batch)

    # compute logloss
    loss_value = loss(logits, tout)
    batch_loss = loss_value.data

    optimizer.zero_grad()
    optimizer_fp.zero_grad()
    optimizer_sf.zero_grad()
    # compute grads for quantized model
    loss_value.backward()

    # get all quantized kernels
    all_kernels = optimizer.param_groups[1]['params']

    # get their full precision backups
    all_fp_kernels = optimizer_fp.param_groups[0]['params']

    # get two scaling factors for each quantized kernel
    scaling_factors = optimizer_sf.param_groups[0]['params']

    for i in range(len(all_kernels)):

        # get a quantized kernel
        k = all_kernels[i]

        # get corresponding full precision kernel
        k_fp = all_fp_kernels[i]

        # get scaling factors for the quantized kernel
        f = scaling_factors[i]
        w_p, w_n = f.data[0], f.data[1]

        # get modified grads
        k_fp_grad, w_p_grad, w_n_grad = get_grads(k.grad.data, k_fp.data, w_p, w_n, t)

        # grad for full precision kernel
        k_fp.grad = Variable(k_fp_grad)

        # we don't need to update the quantized kernel directly
        k.grad.data.zero_()

        # grad for scaling factors
        f.grad = Variable(torch.cuda.FloatTensor([w_p_grad, w_n_grad]).cpu())

    # update all non quantized weights in quantized model
    # (usually, these are the last layer, the first layer, and all batch norm params)
    optimizer.step()

    # update all full precision kernels
    optimizer_fp.step()

    # update all scaling factors
    optimizer_sf.step()

    # update all quantized kernels with updated full precision kernels
    for i in range(len(all_kernels)):

        k = all_kernels[i]
        k_fp = all_fp_kernels[i]
        f = scaling_factors[i]
        w_p, w_n = f.data[0], f.data[1]

        # requantize a quantized kernel using updated full precision weights
        k.data = quantize(k_fp.data, w_p, w_n, t)

    return batch_loss


def optimization_step_fn(model, loss, x_batch, y_batch):
    return optimization_step(
        model, loss, x_batch, y_batch,
        optimizer_list=optimizer_list,
        t=HYPERPARAMETER_T
    )


def train(model, loss, optimization_step_fn,
          train_iterator, val_iterator, n_epochs=30,
          patience=10, threshold=0.01, lr_scheduler=None):
    """
    Train 'model' by minimizing 'loss' using 'optimization_step_fn'
    for parameter updates.
    """

    # collect losses and accuracies here
    all_losses = []

    running_loss = 0.0
    model.train()

    for epoch in range(0, n_epochs):

        # main training loop
        for x_batch, y_batch in train_iterator:

            batch_loss = optimization_step_fn(
                model, loss, x_batch, y_batch
            )
            running_loss += batch_loss
        print("epoch:{}  loss:{}".format(epoch,running_loss))
        running_loss = 0.0

        print("save model!------------------------")
        torch.save(model.state_dict(), './models/qttq/{}/epoch{}_ttq.pkl'.format(bits,epoch))

#-----------------------------------

for i in {12,24,36,48}:
    bits=i
    print("---------{}-----------".format(bits))
    #-------------main function----------
    LEARNING_RATE = 1e-4  # learning rate for all possible weights
    HYPERPARAMETER_T = 0.3  # hyperparameter for quantization

    #----数据导入-------------------------
    from utils import trainloader,testloader
    torch.backends.cudnn.benchmark = True
    #----模型导入-------------------------
    teacher=Resnet18PlusLatent(bits)
    teacher.load_state_dict(torch.load("./models/teacher/T_bit{}.pkl".format(bits)))
    teacher.cuda()
    teacher.train(False)
    model,loss,optimizer=get_model()

    #model.load_state_dict(torch.load('resnet_cifar10_params.pkl'))

    # 全精度的模型kernel拷贝（除了第一层卷积）
    # copy almost all full precision kernels of the model
    all_fp_kernels = [
        Variable(kernel.data.clone(), requires_grad=True)
        for kernel in optimizer.param_groups[1]['params']
    ]
    # all_fp_kernels - kernel tensors of all convolutional layers
    # (with the exception of the first conv layer)

    # 初始量化步骤
    # scaling factors for each quantized layer
    initial_scaling_factors = []

    # these kernels will be quantized
    all_kernels = [kernel for kernel in optimizer.param_groups[1]['params']]

    for k, k_fp in zip(all_kernels, all_fp_kernels):
        # choose initial scaling factors
        w_p_initial, w_n_initial = initial_scales(k_fp.data)
        initial_scaling_factors += [(w_p_initial, w_n_initial)]

        # do quantization
        k.data = quantize(k_fp.data, w_p_initial, w_n_initial, t=HYPERPARAMETER_T)

    # 参数更新
    # optimizer for updating only all_fp_kernels
    optimizer_fp = torch.optim.Adam(all_fp_kernels, lr=LEARNING_RATE)

    # optimizer for updating only scaling factors
    optimizer_sf = torch.optim.Adam([
        Variable(torch.FloatTensor([w_p, w_n]), requires_grad=True)
        for w_p, w_n in initial_scaling_factors
    ], lr=LEARNING_RATE)

    optimizer_list = [optimizer, optimizer_fp, optimizer_sf]
    # 训练
    train(model, loss, optimization_step_fn,trainloader, testloader, n_epochs=100)
    # epoch logloss  accuracy    top5_accuracy time  (first value: train, second value: val)

