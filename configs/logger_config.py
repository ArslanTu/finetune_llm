logger_config = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': f"%(asctime)s %(name)s %(levelname)s: %(message)s",
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        # 其他的 formatter
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': "./log/train_logger.log",
            'level': 'DEBUG',
            'formatter': 'simple',
        },
        # 其他的 handler
    },
    'loggers':{
        # 仅输出到控制台，使用 StreamLogger
        # 输出到控制台，同时写入文件，使用FileLogger
        'StreamLogger': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
        'FileLogger': {
            # 既有 console Handler，还有 file Handler
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
        # 其他的 Logger
    }
}