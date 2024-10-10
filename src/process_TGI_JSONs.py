from dotenv import load_dotenv
import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
import threading
from resource_monitoring import monitor_usage
import pprint

load_dotenv()

# Start monitoring in a separate thread
monitor_thread = threading.Thread(target=monitor_usage)
monitor_thread.daemon = True  # Allow the thread to exit when the main program exits
monitor_thread.start()

def main(rebuild = False):

    file_origin = os.environ.get("FILE_ORIGIN")

    # Path to the input JSON dictionaries for TGI questions
    tgi_input_json_path = file_origin + os.environ.get("TGI_INPUT_PATH")

    # Dump for the processed human-readable JSON questions
    tgi_output_json_path = file_origin + os.environ.get("TGI_OUTPUT_PATH")

    # Path to FAISS byte-serialized mosaic questions for assistant use
    tgi_serialization_path = file_origin + os.environ.get("TGI_SERIAL_PATH")

    mosaics_built = []
    for raw_json in os.listdir(tgi_input_json_path):
        file_path = os.path.join(tgi_input_json_path, raw_json)
        # Ignore any non-JSON files
        if not file_path.endswith('.json'):
            pass

        # Reading the JSON
        with open(file_path, 'r') as file:
            json_dict = json.load(file)

        mosaic_name = raw_json.removesuffix('.json').removesuffix('.JSON')
        if rebuild == True or mosaic_name not in os.listdir(tgi_serialization_path):

            json_dict = json_dict[0]['childHierarchies']
            mosaics_built.append("File: " + raw_json)

            # create JSON object for later incorporation into vectorstore
            question_answer_json = {}
            question_answer_json['mosaic'] = mosaic_name
            question_answer_json['questions'] = []

            # list all categories, subcategories, questions, subquestions, and answers with vert%
            for category in json_dict:
                # run through sub-categories
                for subcategory in category['childHierarchies']:
                    question_string = subcategory['value']
                    saved_question_subcategory = question_string
                    # possible no subcategories exist and we skip straight to question
                    if subcategory['question']:
                        best_answer = 0
                        for answer in subcategory['question']['answers']:
                            if answer['mosaicDTs'] and answer['mosaicDTs'][1]['value'] >= best_answer:
                                answer_string ='\"' + answer['value'] + '\" by ' + str(answer['mosaicDTs'][1]['value']) + '%'
                                best_answer = answer['mosaicDTs'][1]['value']
                        if best_answer > 0:
                            # assemble dictionary to be appended to question list
                            question_answer_element = {}
                            question_answer_element['Question'] = question_string
                            question_answer_element['Top Answer'] = answer_string
                            # append to question list
                            question_answer_json['questions'].append(question_answer_element)
                    # if subcategories exist, we drill deeper
                    elif subcategory['childHierarchies']:
                        for question in subcategory['childHierarchies']:
                            if question['question']:
                                question_string += ' ' + question['question']['value']
                                best_answer = 0
                                for answer in question['question']['answers']:
                                    if answer['mosaicDTs'] and answer['mosaicDTs'][1]['value'] >= best_answer:
                                        answer_string ='\"' + answer['value'] + '\" by ' + str(answer['mosaicDTs'][1]['value']) + '%'
                                        best_answer = answer['mosaicDTs'][1]['value']
                                if best_answer > 0:
                                    # assemble dictionary to be appended to question list
                                    question_answer_element = {}
                                    question_answer_element['Question'] = question_string
                                    question_answer_element['Top Answer'] = answer_string
                                    # append to question list
                                    question_answer_json['questions'].append(question_answer_element)
                                # reset question string
                                question_string = saved_question_subcategory
                            elif question['childHierarchies']:
                                question_string += ' ' + question['value']
                                saved_question_question = question_string
                                for subquestion in question['childHierarchies']:
                                    if subquestion['question']:
                                        question_string += ' ' + subquestion['question']['value']
                                        best_answer = 0
                                        for answer in subquestion['question']['answers']:
                                            if answer['mosaicDTs'] and answer['mosaicDTs'][1]['value'] >= best_answer:
                                                answer_string ='\"' + answer['value'] + '\" by ' + str(answer['mosaicDTs'][1]['value']) + '%'
                                                best_answer = answer['mosaicDTs'][1]['value']
                                        if best_answer > 0 :
                                            # assemble dictionary to be appended to question list
                                            question_answer_element = {}
                                            question_answer_element['Question'] = question_string
                                            question_answer_element['Top Answer'] = answer_string
                                            # append to question list
                                            question_answer_json['questions'].append(question_answer_element)
                                        # reset question string
                                        question_string = saved_question_question
                                # reset question string
                                question_string = saved_question_subcategory
                            else:
                                pass   
                    else:
                        pass

            #JSON Saving block
            tgi_question_json_path = tgi_output_json_path + mosaic_name + ".json"
            with open(tgi_question_json_path, "w") as outfile:
                json.dump(question_answer_json, outfile)

            #Use JSON loader
            question_docs = []
            question_count = len(question_answer_json['questions'])
            for i in range(question_count):
                print('Loading question ' + str(i) + ' of ' + str(question_count) + ' for mosaic ' + mosaic_name)
                loader = JSONLoader(
                    file_path=tgi_question_json_path,
                    jq_schema=f'.questions[{i}]',
                    text_content=False)
                documents = loader.load()
                question_docs.extend(documents)

                #Save into vectorstore and serialize it to bytes

                question_db = FAISS.from_documents(
                    question_docs,
                    OpenAIEmbeddings(model = 'text-embedding-3-small')
                    )
                question_db.save_local(tgi_serialization_path + mosaic_name)

    return mosaics_built  

if __name__ == "__main__":
  print('Rebuild? True/False: ')
  rebuild = input()
  result = main(rebuild)
  print(result)