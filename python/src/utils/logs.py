"""_summary_
    Writer log plots tensorboardX
"""

from tensorboardX import SummaryWriter


class Average: 
    def __init__(self):
        self.reset()


    def update(self, loss, loss_cls, acc, apcer, bpcer, acer):
        self.running_loss += loss
        # self.running_loss_ft += loss_ft
        self.running_loss_cls += loss_cls
        self.running_acc += acc
        self.running_apcer += apcer
        self.running_bpcer += bpcer
        self.running_acer += acer


    def reset(self):
        self.running_loss = 0.
        self.running_acc = 0.
        self.running_loss_cls = 0.
        # self.running_loss_ft = 0.
        self.running_apcer = 0.
        self.running_bpcer = 0.
        self.running_acer = 0.


# init SummaryWriter
def write_tensorboardX(path):
    return SummaryWriter(path)


def write_results(writer: SummaryWriter, metrics, board_loss_every, step, style='train'):
    dict_metric = vars(metrics)
    # Get value
    loss = dict_metric['running_loss'] / board_loss_every
    acc = dict_metric['running_acc'] / board_loss_every
    loss_cls = dict_metric['running_loss_cls'] / board_loss_every
    # loss_ft = dict_metric['running_loss_ft'] / board_loss_every 
    apcer = dict_metric['running_apcer'] / board_loss_every
    bpcer = dict_metric['running_bpcer'] / board_loss_every
    acer = dict_metric['running_acer'] / board_loss_every
    # write plotss
    writer.add_scalar(
        f'{style}/Loss', loss, step)
    writer.add_scalar(
        f'{style}/Acc', acc, step)
    writer.add_scalar(
        f'{style}/Apcer', apcer, step)
    writer.add_scalar(
        f'{style}/Bpcer', bpcer, step)
    writer.add_scalar(
        f'{style}/Acer', acer, step)
    writer.add_scalar(
        f'{style}/Loss_cls', loss_cls, step)
    # writer.add_scalar(
    #     f'{style}/Loss_ft', loss_ft, step)