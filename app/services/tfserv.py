from ..utils import sken_logger
import re
import time
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

logger = sken_logger.get_logger("src/services/models.py")
  
class TfservClient:

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

    def predict(self, inputdata, model, version):
        start_time = time.time()
        text1 = "this is sentence1"
        text2 = "this is sentence2"
        first = self.grpcrequest(inputdata[0], model, version)
        second = self.grpcrequest(inputdata[1], model, version)
        cosine_similarity = tf.keras.losses.cosine_similarity(first, second,axis=1)
        #cosine_similarity = cosineloss(first, second)
        return cosine_similarity.numpy().tolist()

    def predict_1(self, inputdata, model, version):
        start_time = time.time()
        text1 = "this is sentence1"
        text2 = "this is sentence2"
        first = self.grpcrequest(inputdata[0], model, version)
        second = self.grpcrequest(inputdata[1], model, version)
        cosine_similarity = tf.keras.losses.cosine_similarity(first, second,axis=1)
        #cosine_similarity = cosineloss(first, second)
        return cosine_similarity.numpy().tolist()

    def grpcrequest(self, inputdata, model, version):
        
        # Create the PredictRequest and set its values
        req = predict_pb2.PredictRequest()
        req.model_spec.name = model
        req.model_spec.signature_name = 'serving_default'
        tensor = tf.make_tensor_proto(inputdata ,shape=[len(inputdata)],dtype=tf.string)
        req.inputs["input_2"].CopyFrom(tensor)  # Available at /metadata
        response = self.stub.Predict(req, self.REQUEST_TIMEOUT)
        output_tensor_proto = response.outputs["keras_layer_5"]  # Available at /metadata
        shape = tf.TensorShape(output_tensor_proto.tensor_shape)
        result = tf.reshape(output_tensor_proto.float_val, shape)
        return result