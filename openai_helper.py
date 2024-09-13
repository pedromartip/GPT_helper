from openai import OpenAI, OpenAIError
from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np
import datetime
import os
import json
import time
import sys

# Initialize the OpenAI client
client = OpenAI(api_key="sk-p")

def upload_file():
    filename = input("[CLANK]: Enter the filename to upload: ")
    try:
        with open(filename, "rb") as file:
            response = client.files.create(file=file, purpose="assistants")
            print(response)
            print(f"[CLANK]: File uploaded successfully: {response.filename} [{response.id}]")
    except FileNotFoundError:
        print("[CLANK]: File not found. Please make sure the filename and path are correct.")
    except Exception as e:
        print(f"[CLANK]: Error uploading file: {e}")

def list_files():
    try:
        response = client.files.list(purpose="assistants")
        if len(response.data) == 0:
            print("[CLANK]: No files found.")
            return
        for file in response.data:
            created_date = datetime.datetime.utcfromtimestamp(file.created_at).strftime('%Y-%m-%d')
            print(f"{file.filename} [{file.id}], Created: {created_date}")
    except Exception as e:
        print(f"[CLANK]: Error listing files: {e}")

def list_and_delete_file():
    while True:
        response = client.files.list(purpose="assistants")
        files = list(response.data)
        if len(files) == 0:
            print("[CLANK]: No files found.")
            return
        for i, file in enumerate(files, start=1):
            created_date = datetime.datetime.utcfromtimestamp(file.created_at).strftime('%Y-%m-%d')
            print(f"[{i}] {file.filename} [{file.id}], Created: {created_date}")
        choice = input("[CLANK]: Enter a file number to delete, or any other input to return to menu: ")
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(files):
            return
        selected_file = files[int(choice) - 1]
        try:
            client.files.delete(selected_file['id'])
            print(f"[CLANK]: File deleted: {selected_file['filename']}")
        except Exception as e:
            print(f"[CLANK]: Error deleting file: {e}")

def delete_all_files():
    confirmation = input("[CLANK]: This will delete all OpenAI files with purpose 'assistants'.\n Type 'YES' to confirm: ")
    if confirmation == "YES":
        try:
            response = client.files.list(purpose="assistants")
            for file in response['data']:
                client.files.delete(file['id'])
            print("All files with purpose 'assistants' have been deleted.")
        except Exception as e:
            print(f"[CLANK]: Error deleting all files: {e}")
        except OpenAI.OpenAIError as e:
            print(f"[CLANK]: OpenAI API Error: {e}")

def delete_all_vector_stores():
    confirmation = input("[CLANK]: This will delete all OpenAI files with purpose 'assistants'.\n Type 'YES' to confirm: ")
    if confirmation == "YES":
        try:
            vector_stores = client.beta.vector_stores.list()
            while vector_stores['data']:
                for store in vector_stores['data']:
                    client.beta.vector_stores.delete(store.id)
                    print(f"Deleted Vector Store: {store.id}")
                vector_stores = client.beta.vector_stores.list()
            print("[CLANK]: No vector stores found.")
        except Exception as e:
            print(f"[CLANK]: Error deleting vector stores: {e}")
    else:
       print("[CLANK]: Operation cancelled.")

def delete_finetuned_model(model):
    try:
        response = client.models.delete(model)
        if response.deleted == "true":
            print("[CLANK]: The model has been deleted succefully!")
        else:
            print("[CLANK]: Ups! Something went wrong deleting the model.")
    except Exception as e:
        print(f"[CLANK]: Error deleting the model: {e}")

def send_message(message, system_content="You are a helpfull assistant", model="gpt-4o", context=""):
    if not message: 
        print("[CLANK]: No message provided!")
        return ""
    messages = [{"role": "system", "content": system_content}]
    if context: 
        messages.insert(0, {"role": "system", "content": context})
    messages.append({"role": "user", "content": message})
    try:
        completion = client.chat.completions.create(model = model, messages = messages)
        print(f"Response of  {model}: {completion.choices[0].message.content}")
        return completion.choices[0].message.content
    except Exception as e:
        print(f"[CLANK]: Error sending message: {e}")
        return None

def show_available_models():
    try:
        models = client.models.list()  
        print("[CLANK]: Available OpenAI Models:")
        for model in models.data:
            print(f"- {model.id}")
    except Exception as e:
        print(f"[CLANK]: Error showing available models: {e}")

def compare_models(model1, model2, test_file):

    model1_data = []
    model2_data = []
    model1_similarities = []
    model2_similarities = []
    
    try:
        with open(test_file) as f:
            # Deining the model

            ''' WHY COSINE SIMILARITY? (OPEN DICUSSION)
            Here I have a doubt. The fimilarity function we can use is: Cosine comparison, dot product, eucledian distance (L2) and manhattan dist (L1).
            As I know, the cosine comparison is the best choice because the vector is normalised and if we have two outputs and one is a outlier, this one do not modify the result. so the mangitude is not a problem
            because, as more magnitude, more words and that can result with a higher similarity due as more words more probability to match with the desired (embedded) response.
            Otherwise, the dot product is sensible to the magnitude. Then, the eucledian distance is sensible to the magnitude and also is not considering the direction of the vector, the same happend with the Manhattan 
            distance, is nos considering the direction. So for this reason the cosine comparison is probably the best choice.
            '''
            model = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.COSINE) #SBERT model (Not did by Pere SBERT xD) for semantic textual similarity --> https://www.sbert.net/
            for line in f:

                # 1. Extracting the content from JSONL of each aspect: system, user and assistant
                data = json.loads(line)
                system_content = data['messages'][0]['content']
                user_content = data['messages'][1]['content']
                desired_response = data['messages'][2]['content']

                # 2. Sending message and saving response
                response_model1 = send_message(message=user_content, system_content=system_content, model=model1)
                response_model2 = send_message(message=user_content, system_content=system_content, model=model2)

                if response_model1 is None or response_model2 is None:
                    continue

                # 3. Comparing both responses from the desired one in order to evaluate the quality of the LLM response
                embedded_response_desired = model.encode(desired_response)
                embedded_response_model1 = model.encode(response_model1)
                embedded_response_model2 = model.encode(response_model2)

                model1_similarity = model.similarity(embedded_response_model1, embedded_response_desired)
                model2_similarity = model.similarity(embedded_response_model2, embedded_response_desired)
                print(f"Similarity of model 1 = {model1_similarity}")
                print(f"Similarity of model 2 = {model2_similarity}")

                model1_data.append([response_model1, model1_similarity])
                model2_data.append([response_model2, model2_similarity])
                model1_similarities.append([model1_similarity])
                model2_similarities.append([model2_similarity])

        print("[CLANK]: All data collected. Proceeding with results calculation ", end="")            
        for _ in range(3):
            time.sleep(0.33)
            print(". ", end="")  
            sys.stdout.flush() # That is just to be sure that the print is showing inmediatly
        print() 

        mean_model1 = np.mean(model1_similarities)
        mean_model2 = np.mean(model2_similarities)

        print(f'''\n[CLANK]: Here you have the results of the test:\n 
                - Model 1: {mean_model1}\n
                - Model 2: {mean_model2}\n''')
        
        save = input("[CLANK]: Do you want to save the data? Yes (y) - No (Any other input)")
        if save.lower() == "y":
            # Converting the np arrays to lists for JSON
            model1_data_list = model1_data.tolist()
            model2_data_list = model2_data.tolist()

            with open('model1_data.json', 'w') as f1:
                json.dump(model1_data_list, f1)
            with open('model2_data.json', 'w') as f2:
                json.dump(model2_data_list, f2)
            print("[CLANK]: Data saved successfully.")
        else:
            print("[CLANK]: Data was not saved.")
    except FileNotFoundError:  
        print(f"[CLANK]: File not found: {test_file}")
    except OpenAIError as e:
        print(f"[CLANK]: OpenAI API Error: {e}")
    except Exception as e:  
        print(f"[CLANK]: Unexpected Error: {e}")


def main():
    while True:
        print("\n[CLANK]: Hello I'm Clank, your OpenAI LLMs assistant. What you want to do?")
        time.sleep(0.6)
        print("[1] Upload file")
        print("[2] List all files")
        print("[3] List all files and delete one of your choice")
        print("[4] Delete all assistant files (confirmation required)")
        print("[5] Delete all assistant vector stores (confirmation required)")
        print("[6] Delete a fine-tuned model")
        print("[7] Send a message to a model")
        print("[8] Compare two models with a JSONL file")
        print("[9] Show available models")
        print("[0] Exit")
        choice = input("[CLANK]: Enter your choice: ")

        if choice == "1":
            upload_file()
        elif choice == "2":
            list_files()
        elif choice == "3":
            list_and_delete_file()
        elif choice == "4":
            delete_all_files()
        elif choice == "5":
            delete_all_vector_stores()
        elif choice == "6":
            model = input("[CLANK]: Enter your fine-tuned model to delete: ")
            delete_finetuned_model(model)
        elif choice == "7":
            print("[CLANK]: By default the model is the gpt-4o and no context text, what do you want to do?")
            print(" [1] Solely send a message")
            print(" [2] Create the message and choose the model")
            print(" [3] Choose the message context, the model and create the message")
            a = input(" What do you want to do?")
            user_message = input("[CLANK]: Enter your message: ")
            if a == "2" or a == "3": 
                user_model = input("[CLANK]: Enter the model you want to use: ") or "gpt-4o"
            if a == "3": 
                user_context = input("[CLANK]: Enter the context: ")
            response = send_message(message=user_message, model=user_model, context=user_context)
            print(f"[CLANK]: Response --> {response}")
        elif choice == "8":
            print("[CLANK]: Let's start comparing models! Please, carefully input the following information:")
            model1 = input("The first model to compare: ")
            model2 = input("The second model to compare: ")
            test_file = input('''IMPORTANT: The content of the JSONL has to contain data as the following structure: 
                              {"messages": [
                                {"role": "system", "content": "the system content"}, 
                                {"role": "user", "content": "the user content"}, 
                                {"role": "assistant", "content": "the desired response"}
                              ]}\n
                              [CLANK]: Enter the path of the JSONL file with test information: ''')
            compare_models(model1, model2, test_file)
        elif choice == "9":
            show_available_models()
        elif choice == "0":
            print("[CLANK]: See ya!")
            break
        else:
            print("[CLANK]: Invalid choice. Please try again.")

if __name__ == "__main__":
    main()