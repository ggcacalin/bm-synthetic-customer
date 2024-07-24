from dotenv import load_dotenv
import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
import threading
from resource_monitoring import monitor_usage

load_dotenv()

file_origin = os.environ.get("FILE_ORIGIN")

# Path to the input JSON dictionaries for TGI questions
tgi_input_json_path = file_origin + os.environ.get("TGI_INPUT_PATH")

# Dump for the processed human-readable JSON questions
tgi_output_json_path = file_origin + os.environ.get("TGI_OUTPUT_PATH")

# Path to FAISS byte-serialized mosaic questions for assistant use
tgi_serialization_path = file_origin + os.environ.get("TGI_SERIAL_PATH")

# Start monitoring in a separate thread
monitor_thread = threading.Thread(target=monitor_usage)
monitor_thread.daemon = True  # Allow the thread to exit when the main program exits
monitor_thread.start()

def main():
  # Saving raw TGI questions into vectorstore for semantic similarity search
  for count, raw_json in enumerate(os.listdir(tgi_input_json_path)):
    file_path = os.path.join(tgi_input_json_path, raw_json)

    # Reading the JSON
    with open(file_path, 'r') as file:
      json_dict = json.load(file)

    # Mosaic name
    mosaic_name = json_dict['results']['title']
    mosaic_name = mosaic_name.lower()
    mosaic_name = mosaic_name.replace(" ", "_")

    if tgi_serialization_path + mosaic_name not in os.listdir(tgi_serialization_path):
      print(raw_json + " ... " + str(count + 1) + "/" + str(len(os.listdir(tgi_input_json_path))) + " | " + mosaic_name)
      # Data type names
      data_names = []
      for i in range(5):
        data_names.append(json_dict['results']['dataTypes'][i]['label'])
      data_names[0] = 'Population (000)'

      # Question names
      question_names = []
      for i in range(len(json_dict['results']['rows'])):
        question_names.append(json_dict['results']['rows'][i]['label'])

      #Question options
      question_options = []
      for i in range(len(question_names)):
        question_options.append([])
        for j in range(len(json_dict['results']['rows'][i]['values'])):
          question_options[i].append(json_dict['results']['rows'][i]['values'][j]['label'])

      #Condense JSON using previous fields
      question_values = []
      for i in range(len(question_names)):
        condensed_dict = {}
        question_values.append(condensed_dict)
        condensed_dict[question_names[i]] = {}
        for j in range(len(question_options[i])):
          condensed_dict[question_names[i]][question_options[i][j]] = {}
          for k in range(5):
            condensed_dict[question_names[i]][question_options[i][j]][data_names[k]] = json_dict['results']['rows'][i]['values'][j]['data'][k]['vals'][1]

      #Add in the mosaic title
      question_scores_mosaic = {}
      question_scores_mosaic['mosaic'] = json_dict['results']['title'] + ": " + json_dict['results']['filter']['label']
      question_scores_mosaic['questions'] = question_values.copy()

      #Reformat to compute scores
      for i in range(len(question_names)):
        for j in range(len(question_options[i])):
          population = question_scores_mosaic['questions'][i][question_names[i]][question_options[i][j]]['Population (000)']
          index = question_scores_mosaic['questions'][i][question_names[i]][question_options[i][j]]['Index']
          score = population * index / 100
          question_scores_mosaic['questions'][i][question_names[i]][question_options[i][j]] = int(score)

      #Sort questions by highest-scoring answer
      sorted_questions = []
      for question in question_scores_mosaic['questions']:
        for q_name, q_option in question.items():
                sorted_options = dict(sorted(q_option.items(), key=lambda item: item[1], reverse=True))
                sorted_questions.append({q_name: sorted_options})

      #Print and remove every element of sorted_questions that has an empty dictionary at index 0
      removed_count = 0
      for i in reversed(range(len(sorted_questions))):
        if len(sorted_questions[i][next(iter(sorted_questions[i]))]) == 0:
          print(sorted_questions[i])
          del sorted_questions[i]
          removed_count += 1

      print(f"Removed {removed_count} empty questions.")

      # Sort the categories by the highest score in each category
      sorted_question_by_highest = sorted(
          sorted_questions,
          key=lambda q: max(next(iter(q.values())).values()),
          reverse=True
      )
      question_scores_mosaic['questions'] = sorted_question_by_highest

      #JSON Saving block
      tgi_question_json_path = tgi_output_json_path + mosaic_name + ".json"
      with open(tgi_question_json_path, "w") as outfile:
        json.dump(question_scores_mosaic, outfile)

      #Use JSON loader
      question_docs = []
      for i in range(len(question_scores_mosaic['questions'])):
        loader = JSONLoader(
            file_path=tgi_question_json_path,
            jq_schema=f'.questions[{i}]',
            text_content=False)
        documents = loader.load()
        question_docs.extend(documents)

      #Save into vectorstore and serialize it to bytes

      question_db = FAISS.from_documents(
          question_docs,
          OpenAIEmbeddings()
          )
      question_db.save_local(tgi_serialization_path + mosaic_name)

if __name__ == "__main__":
  main()