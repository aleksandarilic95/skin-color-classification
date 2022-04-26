from torch.utils.tensorboard import SummaryWriter
import logging

FORMAT = '[%(asctime)s] %(message)s'
logging.basicConfig(level = logging.DEBUG, format = FORMAT)

class Logger():
    def __init__(self, log_dir = None):
        self.writer = SummaryWriter(log_dir = log_dir)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.NOTSET)


    def add_scalar(self, tag, scalar_value, global_step = None):
        self.writer.add_scalar(tag = tag,
                                scalar_value = scalar_value,
                                global_step = global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step = None):
        self.writer.add_scalars(main_tag = main_tag,
                                 tag_scalar_dict = tag_scalar_dict,
                                 global_step = global_step)

    def add_image(self, tag, img_tensor, global_step = None):
        self.writer.add_image(tag = tag,
                               img_tensor = img_tensor,
                               global_step = global_step)

    def add_images(self, tag, img_tensor, global_step=None):
        self.writer.add_images(tag = tag,
                                img_tensor = img_tensor,
                                global_step = global_step)

    def log_info(self, log):
        self.logger.info(log)
