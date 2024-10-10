import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
load_dotenv()

tgi_serialization_path = os.environ.get("FILE_ORIGIN") + os.environ.get("TGI_SERIAL_PATH")
tgi_output_json_path = os.environ.get('FILE_ORIGIN') + os.environ.get("TGI_OUTPUT_PATH")

with open(tgi_output_json_path + 'Sandra.json', 'r') as file:
    json_dict = json.load(file)

core_questions = ['Age Group', 'Sex', 'Ethnicity']
filtered_QA_pairs = {item['Question']: item['Top Answer'] for item in json_dict['questions'] if item['Question'] in core_questions}

question_db = FAISS.load_local(tgi_serialization_path + 'Sandra', OpenAIEmbeddings(model = 'text-embedding-3-small'), allow_dangerous_deserialization = True)
question_retriever = question_db.as_retriever(search_kwargs = {'k': 3})

questions = question_retriever.invoke("interests and hobbies")
question_return = json.loads(questions[0].page_content)

filtered_QA_pairs[question_return['Question']] = question_return['Top Answer']

print(filtered_QA_pairs)

question_dictionary = {'mosaic': 'Sandra', 'questions': [{'age': '24'}, {'sex': 'male'}]}

question_active_list = question_dictionary['questions']
next_QA_pair = question_active_list[0]
print(next_QA_pair['age'])

if 'Sandra' in os.listdir('files/TGI_Mosaics/Questions In Operation/'):
    print('coaie ce')
else:
    print(os.listdir('files/TGI_Mosaics/Questions In Operation/'))

# Path to plaintext key-insight pairs for assistant use
tgi_insight_path = 'files/' + os.environ.get("TGI_INSIGHT_PATH")

# Path to plaintext set of questions currently in operation to avoid wasteful regeneration
tgi_question_path = 'files/' + os.environ.get("TGI_QUESTION_PATH")

with open(tgi_insight_path + 'Sandra' + '.json', "r") as file:
    keyword_insight_dict = json.load(file)
with open(tgi_question_path + 'Sandra' +'.json', 'r') as file:
    question_dictionary = json.load(file)
previous_keywords = next(reversed(keyword_insight_dict))
previous_question = keyword_insight_dict[previous_keywords]
# Fetching the next question from current list
question_active_list = question_dictionary['questions']
next_QA_pair = question_active_list[0]
print(previous_keywords)
print(previous_question)
for i, question in enumerate(question_active_list):
    print(question['Question'])
    if previous_question == question['Question'] and i+1 < len(question_active_list):
        next_QA_pair = question_active_list[i+1]
        break