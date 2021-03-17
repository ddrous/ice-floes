import logging, logging.handlers
from multiprocessing import Queue
import threading
import sys

FULLDEBUG = 5
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
STATUS = 35 # between the two
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
INTRO = 60

def status(self, message, *args, **kws):
  if self.isEnabledFor(STATUS):
    # Yes, logger takes its '*args' as 'args'.
    self._log(STATUS, message, args, **kws) 

def intro(self, message, *args, **kws):
  self._log(INTRO, message, args, **kws) 

def fulldebug(self, message, *args, **kws):
  self._log(FULLDEBUG, message, args, **kws) 
  
class MyFormater:
  def __init__(self):
    self.regular_formatter = logging.Formatter('%(levelname)-8s - %(message)s')
    self.intro_formatter = logging.Formatter('%(message)s')
  
  def format(self, record):
    if record.levelno == INTRO:
      return self.intro_formatter.format(record)
    else:
      return self.regular_formatter.format(record)
    

class Log:
  def __init__(self, filename, level=WARNING, console_output=False):
    # Logger
    logging.addLevelName(STATUS, "STATUS")
    logging.addLevelName(INTRO, "INTRO")
    logging.addLevelName(FULLDEBUG, "FULLDEBUG")
    logging.Logger.status = status
    logging.Logger.intro = intro
    logging.Logger.fulldebug = fulldebug
    
    self._logger = logging.getLogger('Default')
    self._logger.setLevel('FULLDEBUG')
    
    #formatter = logging.Formatter('%(levelname)-8s - %(message)s')
    formatter = MyFormater()
    file_handler = logging.FileHandler(filename=filename, mode='w+')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    self._logger.addHandler(file_handler)
    if console_output:
      stream = logging.StreamHandler()
      stream.setLevel(STATUS)
      self._logger.addHandler(stream)

    self._log_queue = Queue()
    
    self._log_thread = Log_Thread(self._logger, self._log_queue)
    self._log_thread.start()
    
  def log_description(self, mesh_file, args):
    self._log_queue.put(('INTRO', '## Computations on {}'.format(mesh_file)))
    self._log_queue.put(('INTRO', 'Used args : {}'.format(args)))

  @property
  def _error(self):
    return self._log_thread._error

  def exit(self):
    self._log_queue.put(('STOP', ''))
    self._log_thread.join()
    if self._error:
      sys.exit(_error)


class Log_Thread(threading.Thread):
  def __init__(self, logger, log_queue):
    threading.Thread.__init__(self, daemon=True)
    self.logger = logger
    self.log_queue = log_queue
    self._error = 0

  def run(self):
    while True:
      error, record = self.log_queue.get()
      if error == 'INTRO':
        self.logger.intro(record)
      elif error == 'DEBUG':
        self.logger.debug(record)
      elif error == 'FULLDEBUG':
        self.logger.fulldebug(record)
      elif error == 'INFO':
        self.logger.info(record)
      elif error == 'WARNING':
        self.logger.warning(record)
      elif error == 'STATUS':
        self.logger.status(record)
      elif error == 'ERROR':
        self.logger.error(record)
        _error = 1
      elif error == 'CRITICAL':
        self.logger.critical(record)
        _error = 1
      elif error == 'STOP':
        break
