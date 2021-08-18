from app.services.transformertokenizer import TransformerTokenizer
from ..utils import sken_logger
import re
import time

logger = sken_logger.get_logger("src/services/models.py")
  
class ServiceFactory:

    def __init__(self,tfsclient):
        start_time = time.time()
        logger.info("Testing ServiceFactory client")
        self.model = {}
        self.tfsclient = tfsclient

    
    def add_model(self, modelname):
        start_time = time.time()
        self.model[modelname] = TransformerTokenizer(self.tfsclient, modelname)

    def fetch_response(self, modelname,input):
        start_time = time.time()
        return self.model[modelname].predict(input,modelname)
        
