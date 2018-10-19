# import tensorflow as tf
from tensorboardX import SummaryWriter

# class Logger(object):
#     """Tensorflow Tensorboard logger."""

#     def __init__(self, log_dir):
#         """Initialize summary writer."""
#         self.writer = tf.summary.FileWriter(log_dir)

#     def scalar_summary(self, tag, value, step):
#         """Add scalar summary."""
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#         self.writer.add_summary(summary, step)

class Logger(object):
    """Using tensorboardX such that need no dependency on tensorflow."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)