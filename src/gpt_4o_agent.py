from dotenv import load_dotenv
import os
import pandas as pd
import threading
import json
import process_TGI_JSONs as process_TGI_JSONs
from flask import Flask, request, jsonify
from resource_monitoring import monitor_usage
from random import sample

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
  "tgi_summarization": True,
  "tgi_insight": True,
  "tgi_interrogation": False,
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

# PATH to plaintext mosaic questions for assistant use
tgi_output_json_path = file_origin + os.environ.get("TGI_OUTPUT_PATH")

# Path to FAISS byte-serialized mosaic questions for assistant use
tgi_serialization_path = file_origin + os.environ.get("TGI_SERIAL_PATH")

# Path to plaintext key-insight pairs for assistant use
tgi_insight_path = file_origin + os.environ.get("TGI_INSIGHT_PATH")

# Path to plaintext set of questions currently in operation to avoid wasteful regeneration
tgi_question_path = file_origin + os.environ.get("TGI_QUESTION_PATH")

# Path to IPA FAISS vector store
ipa_faiss_path = file_origin + os.environ.get("IPA_PATH")

# Fixed model parameters
model_temp = 1
model_mt = 1500

# Dynamic model parameters (updated as tools are added)
system_init_prompt = """You are a harmless, helpful, and honest marketing manager talking to a client.
                     Only answer marketing-related questions and avoid saying negative things about your trade or your company.
                     If you are asked to rephrase an answer just present the existing information differently, without using tools again. Keep the same answer format, such as JSON.
                     If you are asked to regenereate an answer to a query after using a tool, use the tool again to find new content to answer the question.
                     When creating prompts for DALL-E, always specify the stylistic parameters: realistic portrait photography with shallow depth of field, shot with a Sigma 50mm f/1.8 lens, high ISO for low-light clarity.
                     """
tool_list = []
function_list = []

# TGI question-based mosaic SUMMARIZATION self-contained tool
if tool_activation["tgi_summarization"]:
  def summarization_get_questions(mosaic_name: str) -> str:
    keywords = "Family, Interests, Money, Media"
    questions_to_return = ""
    # Load core questions on age group, sex, and ethnicity
    with open(tgi_output_json_path + mosaic_name + '.json', 'r') as file:
      json_dict = json.load(file)
    core_questions = ['Age Group', 'Sex', 'Ethnicity']
    return_QA_pairs = {item['Question']: item['Top Answer'] for item in json_dict['questions'] if item['Question'] in core_questions}
    for question in return_QA_pairs:
      questions_to_return += "{'Question': " + question + ", 'Top Answer': " + return_QA_pairs[question] + "}"
      questions_to_return += "\n\n"
    # Load question vectorstore based on mosaic name
    question_db = FAISS.load_local(tgi_serialization_path + mosaic_name, OpenAIEmbeddings(model = 'text-embedding-3-small'), allow_dangerous_deserialization = True)
    question_retriever = question_db.as_retriever(search_kwargs = {'k': 100})
    # Gather most fitting questions for mosaic name and keywords; sample 10 questions from most fitting 100
    question_list = question_retriever.invoke(keywords)
    random_numbers = sample(range(100), 10)
    for i in random_numbers:
      questions_to_return += question_list[i].page_content
      questions_to_return += "\n\n"
    return questions_to_return

  # BaseModel to help assistant know when to call the function
  class SummaryQuestions(BaseModel):
    mosaic_name: str = Field(..., description = "Name of target audience mosaic to find question names and scores for")

  # OpenAI function based on the method, loaded directly into the LLM
  summarization_get_questions_function = {
      "name": "summarization_get_questions",
      "description": "Get the quesstions that create the best short self-description of someone in the target audience.",
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
  summarization_get_questions_tool = StructuredTool.from_function(
      func = summarization_get_questions,
      args_schema = SummaryQuestions,
      description = "Get the quesstions that create the best short self-description of someone in the target audience."
  )

  # Updating the lists of functions and tools and instructing assistant on function use in the prompt
  if summarization_get_questions_tool not in tool_list:
    tool_list.append(summarization_get_questions_tool)
    system_init_prompt += """If the client asks for a description or summary of the target audience mosaic, 
    use your summarization_get_questions tool to get the highest rated answer to relevant questions.
    Present the information as a story about the average individual, giving them a fitting name. For example, you can start by saying \'Meet Jane, a 20 year old student ...\'.
    Load this story into a JSON-like object under the key 'summary'.
    Then formulate a prompt suitable for the DALL-E image generator to create an illustration of the person described by 2-3 key elements of the story. Save this prompt to the JSON object under the 'image_prompt' key.
    If you created portraits of a target audience in the past, tell DALL-E to generate images of the previous person.
    You must return strictly the JSON object with the keys 'summary'and 'image_prompt', formatted properly.
    """
    function_list.append(summarization_get_questions_function)

# TGI question-based INSIGHT GENERATION self-contained tool
if tool_activation["tgi_insight"]:
  k_insights = 31
  def get_insights(mosaic_name: str, search_keywords: str = 'Job, Interests, Consumption, Media Channels') -> str:
    # Load processed JSON dictionary
    with open(tgi_output_json_path + mosaic_name + '.json', 'r') as file:
      json_dict = json.load(file)

    # Extract core insights from processed JSON dictionary
    core_questions = ['Age Group', 'Sex', 'Ethnicity']
    return_QA_pairs = {item['Question']: item['Top Answer'] for item in json_dict['questions'] if item['Question'] in core_questions}

    # Extract relevant non-core insights from vector store
    question_db = FAISS.load_local(tgi_serialization_path + mosaic_name, OpenAIEmbeddings(model = 'text-embedding-3-small'), allow_dangerous_deserialization = True)
    question_retriever = question_db.as_retriever(search_kwargs = {'k': k_insights})

    # If first run and files don't exist:
    if mosaic_name + '.json' not in os.listdir(tgi_question_path):
      # Dealing with questions in operation
      question_list = question_retriever.invoke(search_keywords)
      question_dictionary = {'mosaic': mosaic_name, 'questions': []}
      for i in range(k_insights):
        question_dictionary['questions'].append(json.loads(question_list[i].page_content))
      with open(tgi_question_path + mosaic_name +'.json', 'w') as outfile:
        json.dump(question_dictionary, outfile, indent = 4)
      
      # Dealing with key-insight pair file
      question_return = json.loads(question_list[0].page_content)
      keyword_insight_dict = {search_keywords: question_return['Question']}
      with open(tgi_insight_path + mosaic_name + '.json', "w") as outfile:
        json.dump(keyword_insight_dict, outfile, indent = 4)

      # JSONise the insight question and incorporate in core dicitonary
      return_QA_pairs[question_return['Question']] = question_return['Top Answer']

    # If not first run and the files exist:
    else:
      with open(tgi_insight_path + mosaic_name + '.json', "r") as file:
        keyword_insight_dict = json.load(file)
      # Must resolve whether it's necessary to regenerate question list
      # If keywords are same as previously used, keep question dictionary
      previous_keywords = next(reversed(keyword_insight_dict))
      previous_question = keyword_insight_dict[previous_keywords]
      if previous_keywords == search_keywords:
        with open(tgi_question_path + mosaic_name +'.json', 'r') as file:
          question_dictionary = json.load(file)
        # Fetching the next question from current list
        question_active_list = question_dictionary['questions']
        next_QA_pair = question_active_list[0]
        for i, question in enumerate(question_active_list):
          if previous_question == question['Question'] and i+1 < len(question_active_list):
            next_QA_pair = question_active_list[i+1]
            break
      # If keywords are different, regenerate question dictionary before fetching QA pair
      else:
        question_list = question_retriever.invoke(search_keywords)
        question_dictionary = {'mosaic': mosaic_name, 'questions': []}
        for i in range(k_insights):
          question_dictionary['questions'].append(json.loads(question_list[i].page_content))
        with open(tgi_question_path + mosaic_name +'.json', 'w') as outfile:
          json.dump(question_dictionary, outfile, indent = 4)
        # Fetch first question different from the previous one
        question_active_list = question_dictionary['questions']
        next_QA_pair = question_active_list[0]
        for i, question in enumerate(question_active_list):
          if previous_question != question['Question']:
            next_QA_pair = question_active_list[i]
            break
      # Incorporate QA pair question into keyword-insight dictionary
      keyword_insight_dict[search_keywords] = next_QA_pair['Question']
      if len(keyword_insight_dict) > 10:
        first_key, _ = next(iter(keyword_insight_dict.items()))
        keyword_insight_dict.pop(first_key)
      with open(tgi_insight_path + mosaic_name + '.json', "w") as outfile:
        json.dump(keyword_insight_dict, outfile, indent = 4)
      # Incorporate QA pair into return pairs
      return_QA_pairs[next_QA_pair['Question']] = next_QA_pair['Top Answer']

    # Create super-layer dictionary that will also contain prompt
    return_dict = {'knowledge' : return_QA_pairs}
    return return_dict
  
  # BaseModel to help assistant know when to call the function
  class GetInsights(BaseModel):
    mosaic_name: str = Field(..., description = "Name of target audience mosaic for which to find insights consisting of questions and answers")
    search_keywords: str = Field(..., description = "Keywords in the input message that indicate what insights to look for")

  # OpenAI function based on the method, loaded directly into the LLM
  get_insights_function = {
      "name": "get_insights",
      "description": "Get the insights into the mosaic discussed that match the user's area of interest",
      "parameters": {
          "type": "object",
          "properties": {
              "mosaic_name": {
                  "type": "string",
                  "description": "Name of target audience mosaic for which to find insights consisting of questions and answers"
              },
              "search_keywords": {
                  "type": "string",
                  "description": "Keywords in the input message that indicate what insights to look for"
              }
          },
          "required": ["mosaic_name"]
      }
  }

  # LangChain tool based on the method, loaded into the overarching agent
  get_insights_tool = StructuredTool.from_function(
      func = get_insights,
      args_schema = GetInsights,
      description = "Get the insights into the mosaic discussed that match the user's area of interest"
  )

  # Updating the lists of functions and tools and instructing assistant on function use in the prompt
  if get_insights_tool not in tool_list:
    tool_list.append(get_insights_tool)
    system_init_prompt += """If the client asks for insights, use your get_insights tool with the mosaic name being discussed and any keywords from the input that refer to the type of insights desired, such as 'interests' or 'demographic'.
    Assign the JSON-like object the tool returns as the value of the 'knowledge' key inside a JSON object you will return as the answer.
    From the last question-answer pair, create a natural language version of the given insight. For example, '30% of the target audience has school-age children'.  Save it to the answer JSON object under the 'insight' key.
    From the question-answer pairs in the resulting JSON object, formulate a prompt suitable for the DALL-E image generator to create an illustration of the person described by the insights. Save this prompt to the answer JSON object under the 'image_prompt' key.
    If you previously made up a name for the target audience member you should use it. If you created portraits of a target audience in the past, tell DALL-E to generate images of the previous person.
    You must return strictly the JSON object with the keys 'knowledge','insight', and 'image_prompt' formatted properly.
    """
    function_list.append(get_insights_function)

#TGI question-based mosaic INTERROGATION self-contained tool
if tool_activation["tgi_interrogation"]:
  def interrogate_get_questions(mosaic_name: str, query: str) -> str:
    questions_to_return = ""
    # Load question vectorstore based on mosaic name
    question_db = FAISS.load_local(tgi_serialization_path + mosaic_name, OpenAIEmbeddings(model = 'text-embedding-3-small'), allow_dangerous_deserialization = True)
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
  ipa_db = FAISS.load_local(ipa_faiss_path, OpenAIEmbeddings(model = 'text-embedding-3-small'), allow_dangerous_deserialization = True)
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
    retrieved_memory = ConversationBufferWindowMemory(chat_memory = retrieved_chat_history, k = 4, input_key="input", output_key="output")
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
def syncus():
  data = request.json
  if not data or not all(param in data for param in ("session_id", "mosaic_id", "input_message")):
    return jsonify({"ERROR": "MISSING REQUIRED PARAMETERS"}), 400
  
  session_id = data["session_id"]
  mosaic_id = data["mosaic_id"]
  input_message = data["input_message"]

  response = main(session_id, mosaic_id, input_message)
  return jsonify({"RESPONSE": response})

@agent_app.route('/api/json-update-rebuild-<rebuild>', methods = ['GET'])
def update(rebuild):
  if rebuild.lower() == 'true':
    rebuild_bool = True
  elif rebuild.lower() == 'false':
    rebuild_bool = False
  else:
    return jsonify({"ERROR": "MUST SPECIFY REBUILD True/False"}), 400
  mosaics_built = process_TGI_JSONs.main(rebuild_bool)
  response = {
    "UPDATE_COUNT": len(mosaics_built),
    "MOSAICS_BUILT": mosaics_built
  }
  return jsonify(response)

# Submit command-line message and get response
if __name__ == "__main__":
  session_id = "devtest_session"
  mosaic_id = "Sandra"
  input_message = input()
  while input_message != "nighty night":
    response = main(session_id, mosaic_id, input_message)
    print(response)
    input_message = input()
