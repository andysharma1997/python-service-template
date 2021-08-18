from app.services.servicefactory import ServiceFactory
from app.services.tfservtransformer import TfservTransformerClient
from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field
from typing import Optional, List 
from ..services.tfserv import TfservClient
from ..services.tfserv import TfservClient

router = APIRouter(
    tags=["Inference"],
    responses={404: {"description": "Not found"}},
)

class Input(BaseModel):
    input: List[List[str]] = Field(...,example=[["this is a sentence1", "this is a sentence2"]])
    signature_name:str = Field("serving_default",example="serving_default", description="This is a signature value")

class SingleResponseModel(BaseModel):
    score: float= Field(...,example=0.95)
    similar: bool= Field(...,example=True)

class ResponseModel(BaseModel):
    output: List[SingleResponseModel]

class ResponseModel_2(BaseModel):
    output: List[List[float]]

def create_response(response):
    tmpList = [SingleResponseModel(score=res['score'],similar=res['similar']) for res in response]
    return  ResponseModel(output=tmpList)

client = TfservClient()
ptclient = TfservTransformerClient()
servicefact = ServiceFactory(ptclient)
servicefact.add_model("xlmroberta")

@router.post("/framework/pt/model/{model_name}/version/{version}", response_model=ResponseModel_2)
def read_item(input:Input, model_name: str= Path(..., title="This is a model name"), version:int=Path(..., title="this is the version of the model")):
    inputdata = input.input
    processedData = client.preprocess(inputdata)
    res = servicefact.fetch_response(model_name,inputdata)
    resp = [score for score in res.numpy().tolist()] 
    return ResponseModel_2(output=resp)

@router.post("/framework/tf/model/{model_name}/version/{version}", response_model=ResponseModel)
def read_item(input:Input, model_name: str= Path(..., title="This is a model name"), version:int=Path(..., title="this is the version of the model")):
    inputdata = input.input
    processedData = client.preprocess(inputdata)
    res = client.predict(processedData, model_name, version)
    resp = [{"score":-1*score,"similar":score<-0.5} for score in res] 
    return create_response(resp)

