from dotenv import load_dotenv
import os
import pandas as pd
import threading
import json
from flask import Flask, request, jsonify
from resource_monitoring import monitor_usage

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from pydantic.v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict

load_dotenv()

#Control panel for what tools get loaded
tool_activation ={
  "mosaic_identification": False,
  "title_recommendation": False,
  "tgi_summarization": True,
  "tgi_interrogation": True,
  "tgi_nuggets": True,
  "ipa_retrieval": False
}

# Start monitoring in a separate thread
if not os.path.exists("usage_logs.csv"):
  with open("usage_logs.csv", "w") as f:
        f.write("SEC,CPU,GPU\n")

monitor_thread = threading.Thread(target=monitor_usage)
monitor_thread.daemon = True  # Allow the thread to exit when the main program exits
monitor_thread.start()

# OpenAI credentials
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Path to file origin for all other folders
file_origin = os.environ.get("FILE_ORIGIN")

# Path to FAISS byte-serialized mosaic questions for assistant use
tgi_serialization_path = file_origin + os.environ.get("TGI_SERIAL_PATH")

# Path to IPA FAISS vector store
ipa_faiss_path = file_origin + os.environ.get("IPA_PATH")

# Fixed model parameters
model_temp = 0.5
model_mt = 1500

# Dynamic model parameters (updated as tools are added)
system_init_prompt = """You are a harmless, helpful, and honest marketing manager talking to a client.
                     Only answer marketing-related questions and avoid putting the marketing field or your company in a bad light. """
tool_list = []
function_list = []

#Mosaic identification self-contained tool
if tool_activation["mosaic_identification"]:
  mosaic_title_scores_df = pd.read_csv(os.environ.get("FILE_ORIGIN") + os.environ.get("SQL_CSV_DF"))
  mosaic_db = FAISS.load_local(os.environ.get("FILE_ORIGIN") + os.environ.get("SQL_FAISS"), OpenAIEmbeddings(), allow_dangerous_deserialization = True)
  mosaic_retriever = mosaic_db.as_retriever(search_kwargs = {'k': 100})
  refused_mosaics = []

  def get_mosaics(query: str) -> list:
    # Getting raw table rows, one per mosaic
    document_list = mosaic_retriever.invoke(query)
    mosaic_id_list = []
    for doc in document_list:
      content = doc.page_content
      # Ensuring what was retrieved is a proper table row
      if content[:8] == 'MosaicID':
        # Getting mosaic name and id
        id = int(content[10:content.index('\n')])
        name = content[content.index('Name:'):][6:]
        # Retaining the best matching 3 unique mosaics not already seen
        if id not in mosaic_id_list and name not in refused_mosaics:
          mosaic_id_list.append(id)
          refused_mosaics.append(name)
          if len(mosaic_id_list) == 3:
            break
    # Return the names of the best matching mosaics
    best_mosaic_df = mosaic_title_scores_df.loc[mosaic_title_scores_df['MosaicID'].isin(mosaic_id_list)]
    return best_mosaic_df['Name'].unique().tolist()

  # BaseModel to help assistant know when to call the function
  class GetMosaics(BaseModel):
    query: str = Field(..., description = 'Target audience features that the name field must fit.')

  # OpenAI function based on the method, loaded directly into the LLM
  get_mosaics_function = {
              "name": "get_mosaics",
              "description": "Get the best target audience group descriptions that fit a given list of features.",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "query": {
                          "type": "string",
                          "description": "Target audience features that the name field must fit."
                      }
                  },
                  "required": ["query"]
              }
          }

  # LangChain tool based on the method, loaded into the overarching agent
  get_mosaics_tool = StructuredTool.from_function(
      func = get_mosaics,
      args_schema = GetMosaics,
      description = "Get the best target audience group descriptions that fit a given list of features."
  )

  # Updating the lists of functions and tools and instructing assistant on function use in the prompt
  tool_list.append(get_mosaics_tool)
  system_init_prompt += """Use your get_mosaics tool to recommend the demographic group mosaics that best match the client's description if you are asked about target audiences.
                        If the client is not satisfied with any recommendation, use the function to look up a new set of mosaics.
                        """
  function_list.append(get_mosaics_function)

#Title recommendation and budget allocation self-contained tool
if tool_activation["title_recommendation"]:
  def get_title_scores(mosaic_name: str, budget: float = 10000.00) -> str:
    # Fetch all the titles and scores based on mosaic name matching
    mosaic_df = mosaic_title_scores_df.loc[mosaic_title_scores_df['Name'].str.strip() == mosaic_name.strip()].copy()
    mosaic_df.drop(columns = ['MosaicID', 'TitleCategory', 'Name'], inplace = True)
    # Create budget distribution based on scores
    total_score = mosaic_df['Score'].sum()
    mosaic_df['Budget'] = round(mosaic_df['Score'] / total_score * budget, 2)
    title_score_table = mosaic_df.to_csv(index = False)
    return title_score_table

  # BaseModel to help assistant know when to call the function
  class GetTitleScores(BaseModel):
    mosaic_name: str = Field(..., description = 'Name of target audience mosaic to find score, title, or budget information for.')
    budget: float = Field(..., description = 'The budget to be distributed between titles.')

  # OpenAI function based on the method, loaded directly into the LLM
  get_title_scores_function = {
      "name": "get_title_scores",
      "description": "Get the relevant media titles for a target audience, ranked by score with a distributed budget.",
      "parameters": {
          "type": "object",
          "properties": {
              "mosaic_name": {
                  "type": "string",
                  "description": "Name of target audience mosaic to find score, title, or budget information for."
              },
              "budget": {
                  "type": "number",
                  "description": "The budget to be distributed between titles."
              }
          },
          "required": ["mosaic_name"]
      }
  }

  # LangChain tool based on the method, loaded into the overarching agent
  get_title_scores_tool = StructuredTool.from_function(
      func = get_title_scores,
      args_schema = GetTitleScores,
      description = "Get the relevant media titles for a target audience, ranked by score with a distributed budget."
  )

  # Updating the lists of functions and tools and instructing assistant on function use in the prompt
  if get_title_scores_tool not in tool_list:
    tool_list.append(get_title_scores_tool)
    system_init_prompt += """If the client chooses a mosaic, first ask if they want to know what media titles would make for the most effective advertising campaign, and if they have a budget in mind with a specific currency.
                          If so, use your get_title_scores_tool with the mosaic name and the budget to fetch the title types associated with this mosaic, their scores, and the budget distribution.
                          If no budget is given, use the get_title_scores_tool with the mosaic name only and mention you assumed a value of 10000 when discussing budget allocation.
                          Based on these titles, recommend a focus on the high-scorers and discourage the usage of low-scorers.
                          """
    function_list.append(get_title_scores_function)

# TGI question-based NUGGET GENERATION self-contained tool
if tool_activation["tgi_nuggets"]:
  def nuggets_get_questions(mosaic_name: str) -> str:
    questions_to_return = ""
    question_db = FAISS.load_local(tgi_serialization_path + mosaic_name, OpenAIEmbeddings(), allow_dangerous_deserialization = True)
    question_retriever = question_db.as_retriever(search_kwargs = {'k': 10})
    # Gather most fitting questions for mosaic name and user keywords
    question_list = question_retriever.invoke("Demographics, age, sex, gender, race, ethnicity, nationality, occupation, sector, married, status")
    for question in question_list:
      questions_to_return += question.page_content
      questions_to_return += "\n\n"
    return questions_to_return
  
  # BaseModel to help assistant know when to call the function
  class NuggetQuestions(BaseModel):
    mosaic_name: str = Field(..., description = "Name of target audience mosaic to find question names and scores for")

  # OpenAI function based on the method, loaded directly into the LLM
  nuggets_get_questions_function = {
      "name": "nuggets_get_questions",
      "description": "Get the questions of the mosaic discussed that relate to demographics.",
      "parameters": {
          "type": "object",
          "properties": {
              "mosaic_name": {
                  "type": "string",
                  "description": "Name of target audience mosaic to find question names and scores for"
              }
          },
          "required": ["mosaic_name"]
      }
  }

  # LangChain tool based on the method, loaded into the overarching agent
  nuggets_get_questions_tool = StructuredTool.from_function(
      func = nuggets_get_questions,
      args_schema = NuggetQuestions,
      description = "Get the questions of the mosaic discussed that relate to demographics."
  )

  # Updating the lists of functions and tools and instructing assistant on function use in the prompt
  if nuggets_get_questions_tool not in tool_list:
    tool_list.append(nuggets_get_questions_tool)
    system_init_prompt += """If the client asks for demographic insights or information nuggets, use your nuggets_get_questions tool with the mosaic name being discussed.
                          The numbers represent popularity scores for the response, not numbers of individuals.
                          Create a succint enumeration of demographic features that score the highest in the mosaic, such as age, gender, ethnicity, occupation, and marital status of the average individual in the mosaic.
                          You must not use special characters and you are not allowed to make a bullet point list, write everything in one paragraph.
                          You may infer information on the demographics from what data you have if you can't find what you need in the questions.
                          """
    function_list.append(nuggets_get_questions_function)

# TGI question-based mosaic SUMMARIZATION self-contained tool
if tool_activation["tgi_summarization"]:
  def summarization_get_questions(mosaic_name: str, previous_keywords: str) -> str:
    questions_to_return = ""
    # Load question vectorstore based on mosaic name
    question_db = FAISS.load_local(tgi_serialization_path + mosaic_name, OpenAIEmbeddings(), allow_dangerous_deserialization = True)
    question_retriever = question_db.as_retriever(search_kwargs = {'k': 10})
    # Gather most fitting questions for mosaic name and user keywords
    question_list = question_retriever.invoke(mosaic_name + " " + previous_keywords)
    for question in question_list:
      questions_to_return += question.page_content
      questions_to_return += "\n\n"
    return questions_to_return

  # BaseModel to help assistant know when to call the function
  class SummaryQuestions(BaseModel):
    mosaic_name: str = Field(..., description = "Name of target audience mosaic to find question names and scores for")
    previous_keywords: str = Field(..., description = "Keywords the user previously submitted to get a mosaic recommended")

  # OpenAI function based on the method, loaded directly into the LLM
  summarization_get_questions_function = {
      "name": "summarization_get_questions",
      "description": "Get the questions most related to the mosaic name and audience features given by the user.",
      "parameters": {
          "type": "object",
          "properties": {
              "mosaic_name": {
                  "type": "string",
                  "description": "Name of target audience mosaic to find question names and scores for"
              },
              "previous_keywords": {
                  "type": "string",
                  "description": "Keywords the user previously submitted to get a mosaic recommended"
              }
          },
          "required": ["mosaic_name", "previous_keywords"]
      }
  }

  # LangChain tool based on the method, loaded into the overarching agent
  summarization_get_questions_tool = StructuredTool.from_function(
      func = summarization_get_questions,
      args_schema = SummaryQuestions,
      description = "Get the questions most related to the mosaic name and audience features given by the user."
  )

  # Updating the lists of functions and tools and instructing assistant on function use in the prompt
  if summarization_get_questions_tool not in tool_list:
    tool_list.append(summarization_get_questions_tool)
    system_init_prompt += """If the client asks for a description or summary of the target audience mosaic, use your summarization_get_questions tool with the same query from get_mosaics and the mosaic name.
                          The numbers represent popularity scores for the response, not numbers of individuals.
                          Give the description as a story about an individual from the group, not an itemized list.
                          """
    function_list.append(summarization_get_questions_function)

#TGI question-based mosaic INTERROGATION self-contained tool
if tool_activation["tgi_interrogation"]:
  def interrogate_get_questions(mosaic_name: str, query: str) -> str:
    questions_to_return = ""
    # Load question vectorstore based on mosaic name
    question_db = FAISS.load_local(tgi_serialization_path + mosaic_name, OpenAIEmbeddings(), allow_dangerous_deserialization = True)
    question_retriever = question_db.as_retriever(search_kwargs = {'k': 3})
    # Gather most fitting questions for the user's query
    question_list = question_retriever.invoke(query)
    for question in question_list:
      questions_to_return += question.page_content
      questions_to_return += "\n\n"
    return questions_to_return

  # BaseModel to help assistant know when to call the function
  class InterrogateQuestions(BaseModel):
    mosaic_name: str = Field(..., description = "Name of target audience mosaic to find question names and scores for")
    query: str = Field(..., description = "The question the user is asking about the mosaic")

  # OpenAI function based on the method, loaded directly into the LLM
  interrogate_get_questions_function = {
      "name": "interrogate_get_questions",
      "description": "Get the characteristics of the mosaic that best match the user's query.",
      "parameters": {
          "type": "object",
          "properties": {
              "mosaic_name": {
                  "type": "string",
                  "description": "Name of target audience mosaic to find question names and scores for"
              },
              "query": {
                  "type": "string",
                  "description": "The question the user is asking about the mosaic"
              }
          },
          "required": ["mosaic_name", "query"]
      }
  }

  # LangChain tool based on the method, loaded into the overarching agent
  interrogate_get_questions_tool = StructuredTool.from_function(
      func = interrogate_get_questions,
      args_schema = InterrogateQuestions,
      description = "Get the characteristics of the mosaic that best match the user's query."
  )

  # Updating the lists of functions and tools and instructing assistant on function use in the prompt
  # TODO: Make the interests bit more explicit
  if interrogate_get_questions_tool not in tool_list:
    tool_list.append(interrogate_get_questions_tool)
    system_init_prompt += """If the client asks more detailed questions about the demographics or interests within a target audience, use your interrogate_get_questions tool with the exact mosaic name provided and the user's query.
                          The numbers represent popularity scores for the response, not numbers of individuals. Do not mention the scores.
                          Regardless of how the question is phrased, act like you are an individual from the target audience mosaic and answer as if the question was addressed to you. Do not mention the name of the mosaic you belong to.
                          You may use knowledge outside the context provided by the characteristic scores from the documents to supplement your answer.
                          """
    function_list.append(interrogate_get_questions_function)

# IPA knowledge retrieval tool
if tool_activation["ipa_retrieval"]:
  ipa_db = FAISS.load_local(ipa_faiss_path, OpenAIEmbeddings(), allow_dangerous_deserialization = True)
  ipa_retriever = ipa_db.as_retriever(search_kwargs = {'k': 3})

  # Deal with the tool
  ipa_retriever_tool = create_retriever_tool(
      ipa_retriever,
      "ipa_search",
      "Look up marketing expertise that helps answer the question of the user. For any marketing related questions, you must use this tool!"
  )

  if ipa_retriever_tool not in tool_list:
    tool_list.append(ipa_retriever_tool)
    system_init_prompt += """Use your ipa_search tool to look up marketing expertise that may enrich your answer. For any marketing related questions, you must use this tool!
                          """

  # Deal with the function
  ipa_retriever_function = convert_to_openai_function(ipa_retriever_tool)

  if ipa_retriever_function not in function_list:
    function_list.append(ipa_retriever_function)


# Initialize LLM
tools = tool_list
llm = ChatOpenAI(temperature = model_temp, max_tokens = model_mt, model_name = "gpt-4o")
llm_with_tools = llm.bind(
    functions = function_list
)

# Method for loading existing memory or creating a new memory file
def load_memory(session_id):
  memory_path = os.environ.get("FILE_ORIGIN") + os.environ.get("HISTORY_PATH") + str(session_id) + '.json'
  # Loads the memory if it already exists
  if os.path.isfile(memory_path):
    with open(memory_path, 'r') as file:
      retrieved_messages = messages_from_dict(json.load(file))
    retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
    retrieved_memory = ConversationBufferWindowMemory(chat_memory = retrieved_chat_history, k = 5, input_key="input", output_key="output")
    return retrieved_memory
  # Creates a fresh memory and a JSON for it to be dumped at the end
  else:
    with open(memory_path, 'w') as file:
      json.dump([], file)
    return ConversationBufferWindowMemory(k = 5, input_key="input", output_key="output")

# Method to submit a message, get a response, and have the interaction recorded in history
def submit_message(session_id, mosaic_id, input_message):
  # Deal with memory
  memory_path = os.environ.get("FILE_ORIGIN") + os.environ.get("HISTORY_PATH") + str(session_id) + '.json'
  memory = load_memory(session_id)

  # Create user prompt and combine with system for full version
  user_init_prompt = "All my target audience questions are related to the following target mosaic: " + mosaic_id + "\n"
  user_init_prompt += "Here is what I want: {}"

  prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system_init_prompt),
          MessagesPlaceholder(variable_name="chat_history"),
          ("user", user_init_prompt.format("{input}")),
          MessagesPlaceholder(variable_name="agent_scratchpad"),
      ],
  )

  # Initialize agent
  agent = (
      {
          "input": lambda x: x["input"],
          "agent_scratchpad": lambda x: format_to_openai_function_messages(
              x["intermediate_steps"]
          ),
          "chat_history": lambda x: x["chat_history"],
      }
      | prompt
      | llm_with_tools
      | OpenAIFunctionsAgentOutputParser()
  )

  agent_executor = AgentExecutor(agent=agent,
                                tools=tools,
                                memory=memory,
                                verbose=True,
                                handle_parsing_errors=True,
                                return_intermediate_steps=True)
  response = agent_executor.invoke({"input": input_message, "chat_history": memory.buffer_as_messages})
  extracted_messages = memory.chat_memory.messages
  with open(memory_path, 'w') as file:
    json.dump(messages_to_dict(extracted_messages), file, indent = 4)
  return response['output']

# Main method to interact with the agent and retrieve the response
def main(session_id, mosaic_id, input_message):
  response = submit_message(session_id, mosaic_id, input_message)
  return response

# Deployment related API setup
agent_app = Flask(__name__)
@agent_app.route('/api/syncus', methods = ['POST'])
def process():
  data = request.json
  if not data or not all(param in data for param in ("session_id", "mosaic_id", "input_message")):
    return jsonify({"ERROR": "MISSING REQUIRED PARAMETERS"}), 400
  
  session_id = data["session_id"]
  mosaic_id = data["mosaic_id"]
  input_message = data["input_message"]

  response = main(session_id, mosaic_id, input_message)
  return jsonify({"RESPONSE": response})

# Submit command-line message and get response
if __name__ == "__main__":
  agent_app.run(port = 80)
