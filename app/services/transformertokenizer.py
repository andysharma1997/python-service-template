from app.services.tfservtransformer import TfservTransformerClient
from app.services.tfserv import TfservClient
from ..utils import sken_logger
import re
import time
from transformers import AutoTokenizer

logger = sken_logger.get_logger("src/services/models.py")
  
class TransformerTokenizer:

    def __init__(self, tfsclient:TfservTransformerClient, modelname):
        start_time = time.time()
        logger.info("Testing TransformerTokenizer client")
        self.modelname = modelname
        self.tfsclient = tfsclient
        self.tokenizer = AutoTokenizer.from_pretrained("salesken/xlm-roberta-base-finetuned-mnli-cross-lingual-transfer")
        

    
    def predict(self, inputdata, modelname):
        start_time = time.time()
        processeddata = self.tfsclient.preprocess(inputdata)
        inp=self.tokenizer.encode_plus('This is a testtt, hah! reaaly cool :)','This is a testtt, hah! reaaly cool :)',
                          max_length=128, 
                          padding='max_length',
                          truncation=True, 
                          return_tensors='tf')
        return self.tfsclient.predict(inp,modelname)
