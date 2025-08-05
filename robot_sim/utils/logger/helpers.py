import os
import os.path as osp 
import json
import socket 
import datetime 

LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

def get_config(config_json, log_stream_level, log_file_level, log_dir):
    with open(config_json, 'r') as f:
        config = json.load(f)

    set_log_level(config, log_stream_level, log_file_level)
    set_log_dir(config, log_dir)
    
    return config

def set_log_level(config, log_stream_level, log_file_level):
    log_stream_level = log_stream_level.upper()
    log_file_level = log_file_level.upper()
    assert log_stream_level.upper() in get_log_levels(), ValueError(f"Log-level for streaming should be one of {LOG_LEVELS}")
    assert log_file_level.upper() in get_log_levels(), ValueError(f"Log-level for file should be one of {LOG_LEVELS}")

    config['handlers']['stream']['level'] = log_stream_level
    config['root']['level'] = log_file_level

def set_log_dir(config, log_dir):
    if log_dir == None:
        hostname = socket.gethostname()

        log_dir = f'/home/{hostname}/logs'
        current_date = datetime.datetime.now()
        year = current_date.year
        month = current_date.month
        day = current_date.day 
        hour = current_date.hour
        
        log_dir = osp.join(log_dir, f"{year}_{month}_{day}", str(hour))
        
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
    
    assert log_dir != None, ValueError(f"Log directory should be assigned, not {log_dir}")
    
    config['handlers']['file']['filename'] = \
                        osp.join(log_dir, config['handlers']['file']['filename'])

def get_log_levels():
    return LOG_LEVELS