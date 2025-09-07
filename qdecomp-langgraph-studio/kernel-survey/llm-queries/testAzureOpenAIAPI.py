
from langchain_openai import AzureChatOpenAI, ChatOpenAI 
from langchain.prompts import ChatPromptTemplate
import os
import langchain
from langchain_core.runnables.utils import ConfigurableField


print(langchain.__version__)

#llm = AzureChatOpenAI(
#  api_key=os.getenv("OPENAI_API_KEY"),
#  azure_endpoint="https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com",
#  api_version="2025-04-01-preview",
#  temperature=1,
#  top_p=1,
#  model_name="gpt-5-mini",
#)

llm = AzureChatOpenAI(
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
  azure_endpoint="https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com",
  api_version="2025-04-01-preview",
  temperature=1,
  top_p=1,
  model_name="gpt-5-mini",
).configurable_fields(
    model_name=ConfigurableField(
        id="model",
        is_shared=True,
    ),
    temperature=ConfigurableField(
        id="temp",
        is_shared=True,
    ),
    top_p=ConfigurableField(
        id="top_p",
        is_shared=True,
    ),
    azure_endpoint=ConfigurableField(
        id="provider_url",
        is_shared=True,
    ),
    api_key=ConfigurableField( 
        id="provider_api_key",
        is_shared=True,
    )
).configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="azure",
    openai=ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        top_p=0.1,
        model_name="openai/o3-mini",
    ).configurable_fields(
    model_name=ConfigurableField(
        id="opr_model",
        is_shared=True,
    ),
    temperature=ConfigurableField(
        id="opr_temp",
        is_shared=True,
    ),
    top_p=ConfigurableField(
        id="opr_top_p",
        is_shared=True,
    ),
    openai_api_base=ConfigurableField(
        id="opr_provider_url",
        is_shared=True,
    ),
    openai_api_key=ConfigurableField( 
        id="opr_provider_api_key",
        is_shared=True,
    )
)
)


#llm = ChatOpenAI(
#  api_key=os.getenv("OPENAI_API_KEY"),
#  base_url="https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com/openai/responses",
#  temperature=1,
#  top_p=1,
#  model_name="gpt-5-mini",
#  #api_version="2025-04-01-preview",
#  openai_organization="&api-version=2025-04-01-preview"
#)

'''
llm = ChatOpenAI(
  openai_api_key=os.getenv("OPENAI_API_KEY"),
  openai_api_base="https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview",
  temperature=1,
  top_p=1,
  model_name="gpt-5-mini",
)
'''

prompt = ChatPromptTemplate.from_messages([
    ("human", 
    #"Explain how to solve the following math problem: 12x + 5 = 12x^2 + sin(x) using fourier series."
    "What is the capital of France?"
     )
])
chain = prompt | llm.with_config(configurable={"llm": "openai", 
                                               #"provider_url":"https://openrouter.ai/api/v1", 
                                               "opr_provider_api_key": os.getenv("OPENAI_API_KEY"), 
                                               "opr_model": "google/gemini-2.5-flash-lite", 
                                               "opr_temp":0.2, 
                                               "opr_top_p":0.1,
                                               })

result = chain.invoke({})

print(result)