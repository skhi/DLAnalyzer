import logging
import sys

log = logging.getLogger("log.framework")
log.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(name)s %(threadName)-11s %(levelname)-10s %(message)s")

# Log to file
filehandler = logging.FileHandler("log.txt", "w")
filehandler.setLevel(logging.DEBUG)
filehandler.setFormatter(formatter)
log.addHandler(filehandler)

# Log to stdout too
streamhandler = logging.StreamHandler() # (sys.stdout)
streamhandler.setLevel(logging.INFO)
streamhandler.setFormatter(formatter)
log.addHandler(streamhandler)
#sys.stdout = log.info
