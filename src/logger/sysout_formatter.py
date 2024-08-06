import logging
from logging import Formatter


class SysoutFormatter(Formatter):
    """A custom log formatter that adds a field with the initial of the log's level."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a custom log record by adding a new field with the initial of the logging letter.

        Args:
            record: A logging record.

        Returns: The logging record with the updated field.
        """
        log_message = super(SysoutFormatter, self).format(record)
        return log_message
