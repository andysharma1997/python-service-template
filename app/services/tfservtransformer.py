from ..utils import sken_logger
import re
import time
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

logger = sken_logger.get_logger("src/services/models.py")
  
class TfservTransformerClient:

    def __init__(self):
        start_time = time.time()
        logger.info("Testing tfserv client")
        # Optional: define a custom message lenght in bytes
        MAX_MESSAGE_LENGTH = 20000000
        # Optional: define a request timeout in seconds
        self.REQUEST_TIMEOUT = 5
        # Open a gRPC insecure channel
        channel = grpc.insecure_channel(
                "35.200.235.122:8500",
                options=[
                    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
                ],
        )
        # Create the PredictionServiceStub
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    def preprocess(self,inputdata):
        first = [row[0] for row in inputdata]
        second = [row[1] for row in inputdata]
        return [first, second]

    def predict(self, inputdata, model):
        start_time = time.time()
        text1 = "this is sentence1"
        text2 = "this is sentence2"
        first = self.grpcrequest(inputdata, model)
        #second = self.grpcrequest(inputdata[1], model, version)
        #cosine_similarity = tf.keras.losses.cosine_similarity(first, second,axis=1)
        #cosine_similarity = cosineloss(first, second)
        return first#cosine_similarity.numpy().tolist()

    def grpcrequest(self, inputdata, model):
        
        # Create the PredictRequest and set its values
        req = predict_pb2.PredictRequest()
        req.model_spec.name = model
        req.model_spec.signature_name = 'serving_default'
        tensor1 = tf.make_tensor_proto(inputdata['input_ids'])
        tensor2= tf.make_tensor_proto(inputdata['attention_mask'])
        req.inputs["input_ids"].CopyFrom(tensor1)  # Available at /metadata
        req.inputs["attention_mask"].CopyFrom(tensor2)  # Available at /metadata
        response = self.stub.Predict(req, self.REQUEST_TIMEOUT)
        logger.info(response.outputs)
        output_tensor_proto = response.outputs["logits"]  # Available at /metadata
        shape = tf.TensorShape(output_tensor_proto.tensor_shape)
        result = tf.reshape(output_tensor_proto.float_val, shape)
        logger.info(result)
        return result