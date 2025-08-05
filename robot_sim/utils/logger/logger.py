import os
import os.path as osp
import logging
import logging.config
import warnings
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[0]

from robot_sim.utils.logger.helpers import get_config

class Logger:
    def __init__(self, name=None, log_stream_level="DEBUG", log_file_level="DEBUG", \
                        log_dir=None, config_json=osp.join(ROOT, 'data/logging.json')):
    
        self._name = name 
        self._log_stream_level = log_stream_level 
        self._log_file_level = log_file_level 
        self._log_dir = log_dir 
        self._config_json = config_json
        self._logger = None
        
        self._set()
        self.info(f"Logger has been set with ")
        
    def _set(self):
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            config = get_config(self._config_json, self._log_stream_level, \
                                self._log_file_level, self._log_dir)
            logging.config.dictConfig(config)
            self._logger = logging.getLogger(self._name)
        except Exception as e:
            warnings.warn(f"Cannot define logger: {e}")
            self._logger = None
    
    def debug(self, msg):
        if self._logger != None:
            self._logger.debug(msg)
        
    def info(self, msg):
        if self._logger != None:
            self._logger.info(msg)
    
    def warning(self, msg):
        if self._logger != None:
            self._logger.warning(msg)
    
    def error(self, msg):
        if self._logger != None:
            self._logger.error(msg)
    
    def critical(self, msg):
        if self._logger != None:
            self._logger.critical(msg)
    
    def try_except_log(self, func, msg="", post_action=None, exit=False):
        try:
            if msg == 'In the post-action, ':
                self._logger.info("Post-action runs after rasing error")
            func()
        except Exception as error_msg:
            error_type = type(error_msg).__name__
            if len(msg) != 0:
                error_msg = msg + ", and "  + str(error_msg)
            else:
                error_msg = str(error_msg)
            self._logger.error(f" [{self.try_except_log.__name__}] {error_type}: {error_msg}")
            
            if post_action is not None:
                self.try_except_log(post_action, msg='In the post-action, ')
            if error_type in __builtins__:
                if not exit:
                    raise __builtins__[error_type](error_msg)
                else:
                    sys.exit(1)
            
        
