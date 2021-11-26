import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s]: %(levelname)s - %(message)s')


class Logger(object):

    def __init__(self, actuator: str = 'unknown'):
        self.logger = logging.getLogger(actuator)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warn(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def info_train(self, task_name, device, train_cnt, accuracy, loss, current_epoch=0, total_epoch=0):
        self.logger.info(
            (f"[{current_epoch:0>3d}/{total_epoch:0>3d}] " if current_epoch and total_epoch else f"") +
            f"Train '{task_name}' on {device} with {train_cnt:,} images, " +
            f"accuracy: {accuracy:.2%}, loss: {loss:.4f}."
        )

    def info_validation(self, task_name, query_cnt, gallery_cnt, cmc, mAP) -> None:
        self.logger.info(
            """Validation '{}' with {:,} query images on {:,} gallery images:
            |- Rank-1 :  {:.2%}
            |- Rank-3 :  {:.2%}
            |- Rank-5 :  {:.2%}
            |- Rank-10 : {:.2%}
            |- mean AP : {:.2%}
            """.format(task_name, query_cnt, gallery_cnt, cmc[0], cmc[2], cmc[4], cmc[9], mAP)
        )
