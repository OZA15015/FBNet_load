import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import torchvision.datasets as datasets

from general_functions.dataloaders import get_loaders, get_test_loader
from general_functions.utils import get_logger, weights_init, create_directories_from_list
import fbnet_building_blocks.fbnet_builder as fbnet_builder
from architecture_functions.training_functions import TrainerArch
from architecture_functions.config_for_arch import CONFIG_ARCH
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser("architecture")

parser.add_argument('--architecture_name', type=str, default='', \
                    help='You can choose architecture from the fbnet_building_blocks/fbnet_modeldef.py')
args = parser.parse_args()

def main():
    manual_seed = 1
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.benchmark = True
    
    create_directories_from_list([CONFIG_ARCH['logging']['path_to_tensorboard_logs']])
    
    logger = get_logger(CONFIG_ARCH['logging']['path_to_log_file'])
    writer = SummaryWriter(log_dir=CONFIG_ARCH['logging']['path_to_tensorboard_logs'])

    #### DataLoading
    train_loader = get_loaders(1.0, CONFIG_ARCH['dataloading']['batch_size'],
                               CONFIG_ARCH['dataloading']['path_to_save_data'],
                               logger)
    valid_loader = get_test_loader(CONFIG_ARCH['dataloading']['batch_size'],
                                   CONFIG_ARCH['dataloading']['path_to_save_data'])
    
    #### Model
    arch = args.architecture_name
    model = fbnet_builder.get_model(arch, cnt_classes=10).cuda()
    #model.load_state_dict(torch.load("architecture_functions/logs/best_model.pth", map_location=torch.device('cuda')))
    #checkpoint = torch.load("architecture_functions/logs/best_model.pth", map_location=torch.device('cuda'))
    
    #state_dict = torch.load("architecture_functions/logs/best_model.pth", map_location="cuda")
    state_dict = torch.load("/home/oza/pre-experiment/speeding/FBNet/architecture_functions/logs/best_model.pth", map_location="cuda")
    #state_dict = torch.load("/home/oza/pre-experiment/speeding/test_dist/logs/test_FBnetA0826/best.pth.tar")['state_dict']
    #state_dict = torch.load("/home/oza/pre-experiment/speeding/testFB/FBNet/architecture_functions/logs/fbnet_a/best_model.pth", map_location="cuda")
    #state_dict = torch.load("/home/oza/pre-experiment/speeding/testFB/distiller/examples/classifier_compression/logs/2020.08.29-035310/best.pth.tar", map_location="cuda")['state_dict']
    #state_dict = torch.load("/home/oza/pre-experiment/speeding/testFB/FBNet/architecture_functions/logs/best_model.pth", map_location="cuda")
    #state_dict = torch.load("/home/oza/pre-experiment/speeding/testFB/FBNet/architecture_functions/logs/fbnet_a/best_model.pth", map_location="cuda")
    if "model_ema" in state_dict and state_dict["model_ema"] is not None:
        state_dict = state_dict["model_ema"]

    ret = {}
    for name, val in state_dict.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        #print(name)
        ret[name] = val 


    # ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD  = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

    # トレーニングデータをダウンロード
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
 
    # テストデータをダウンロード
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)
    

    test_data = datasets.CIFAR10(root='./data', train=False,
                                 download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=4,
                                              shuffle=False, num_workers=16)
    

    model.load_state_dict(ret)
    model.eval()
    correct = 0
    total = 0
    topk=(1,)
    
    '''
    for data in testloader:
        images, labels = data                   
        images  = images.to('cuda')             
        labels = labels.to('cuda')              
        outputs = model(images) 
    
    
    maxk = max(topk)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if labels.ndimension() > 1:
        labels = labels.max(1)[1]
     
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
     
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / 10000))
     
    print(res)
    '''
    i = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images  = images.to('cuda')
            labels = labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #print(total)
            #print(correct)
            i += 1
            print(i)
    print(total)
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    quit()

    model = model.apply(weights_init)
    model = nn.DataParallel(model, [0])

    #### Loss and Optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=CONFIG_ARCH['optimizer']['lr'],
                                momentum=CONFIG_ARCH['optimizer']['momentum'],
                                weight_decay=CONFIG_ARCH['optimizer']['weight_decay'])
    criterion = nn.CrossEntropyLoss().cuda()
    
    #### Scheduler
    if CONFIG_ARCH['train_settings']['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=CONFIG_ARCH['train_settings']['milestones'],
                                                    gamma=CONFIG_ARCH['train_settings']['lr_decay'])  
    elif CONFIG_ARCH['train_settings']['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=CONFIG_ARCH['train_settings']['cnt_epochs'],
                                                               eta_min=0.001, last_epoch=-1)
    else:
        logger.info("Please, specify scheduler in architecture_functions/config_for_arch")
        
    
    #### Training Loop
    trainer = TrainerArch(criterion, optimizer, scheduler, logger, writer)
    trainer.train_loop(train_loader, valid_loader, model) 
    
if __name__ == "__main__":
    main()
