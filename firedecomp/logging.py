import logging


def fix_work(self, message, *args, **kws):
    if self.isEnabledFor(60):
        # Yes, logger takes its '*args' as 'args'.
        self._log(60, message, args, **kws)


class FixWorkFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != 60
