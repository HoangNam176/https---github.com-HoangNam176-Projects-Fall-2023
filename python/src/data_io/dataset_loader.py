from torch.utils.data import DataLoader
from src.data_io.dataset_folder import DatasetFolderFT, Dataset_CelebA_Spoof
from src.data_io import transform as trans
from easydict import EasyDict


def get_train_loader(conf):
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.Resize(conf.size),
        trans.RandomResizedCrop(size=tuple(conf.input_size),
                                scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.2,
                        contrast=0.2, saturation=0.2, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    train_root_path = conf.train_root_path
    train_label_path = conf.train_label_path
    trainset = Dataset_CelebA_Spoof(train_root_path, train_label_path, train_transform,
                            None, conf.ft_width, conf.ft_height)
    print(len(trainset))
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2)
    return train_loader


def get_val_loader(conf):
    val_transform = trans.Compose([
        trans.ToPILImage(),
        trans.Resize(conf.size),
        trans.ToTensor()
    ])
    
    val_root_path = conf.val_root_path
    val_label_path = conf.val_label_path
    valset = Dataset_CelebA_Spoof(val_root_path, val_label_path, val_transform,
                            None, conf.ft_width, conf.ft_height)
    print(len(valset))
    val_loader = DataLoader(
        valset,
        batch_size=conf.val_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2)
    return val_loader


if __name__ == '__main__':
    
    conf = EasyDict()

    # ----------------------training---------------
    conf.lr = 1e-3
    conf.size = 80
    # [9, 13, 15]
    conf.milestones = [15, 120, 26]  # down learing rate
    conf.gamma = 0.1
    conf.epochs = 40
    conf.momentum = 0.9
    conf.batch_size = 128
    conf.val_batch_size = 32

    # model
    conf.num_classes = 2
    conf.input_channel = 3
    conf.embedding_size = 128

    # dataset
    conf.train_root_path = '/home/hoang/Documents/work/Data/Face_Anti_Spoofing/CelebA_Spoof'
    conf.val_root_path = '/home/hoang/Documents/work/Data/Face_Anti_Spoofing/CelebA_Spoof'
    # label
    conf.train_label_path = '/home/hoang/Documents/work/Data/Face_Anti_Spoofing/CelebA_Spoof/metas/intra_test/train_label.txt'
    conf.val_label_path = '/home/hoang/Documents/work/Data/Face_Anti_Spoofing/CelebA_Spoof/metas/intra_test/test_label.txt'
    
    # save file path
    conf.snapshot_dir_path = './saved_logs/snapshot'

    # log path
    conf.log_path = './saved_logs/jobs'
    # tensorboard
    conf.board_loss_every = 10
    # save model/iter
    conf.save_every = 2000
    
    get_val_loader(conf)

