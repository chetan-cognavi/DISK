import pandas as pd
import warnings
import random, requests, re
from mangum import Mangum
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, HTTPException, Request
from pymongo import MongoClient
import numpy as np


warnings.filterwarnings("ignore")

# Replace this with your actual secret token
SECRET_TOKEN = "cognavi_video_ai_m?6m2dB7Q9NipXPWEIdBhBdQekTBx37fqaLl5NzaBThQDmMlw7!f!lUORT?Arlppx2FE/PHP7k8=oQA3pFdvT=Ll6D3qgd-G4KypM19v0Eg!Xw?UCCdXIt7?Xsyw7CBOkFd00XNRwnA?=Cq=x47CwPtqN0nQ5vvSSgZrdF0x1KHA-=CUAFKnIAk0KINcKJ5Q3nI=Bp0XH9lwBe4vZi0HA!Kf5sIxSffLvDoyORnFH3qj1V="

app = FastAPI()
handler = Mangum(app)

def list_to_string(lst):
    return str(', '.join(map(str, lst)))

def break_down_feedback(trait):
    answerstr2 = """Generate short and concise overall feedback in 2 lines to student based on following DISC personality trait and Score. 
    Following score is out of 33.33 and  Strictly do not include any numerical value in response.
    Provide only where student is lagging and how they can improve their personality.
    I repeat, Strictly do not include any numerical score in feedback response. Strictly Start the response with You.
""" + "\n\n" + str(trait)

    headers = {
    'Authorization': 'Bearer ' + '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',
    'Content-Type': 'application/json',
}

    json_data = {
        'model': 'gpt-3.5-turbo-16k',
        'temperature':0.2,
        'messages': [
            {
                'role': 'user',
                'content': answerstr2
            }
        ],
        }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data)

    # Process the response
    if response.status_code == 200:
        res=response.json()
        res=res['choices'][0]['message']['content']
        return res
    else:
        print(f"Failed with status code: {response.status_code}")
        return response.text



class PredictReq(BaseModel):
    student_id: str
    disc_payload: dict

class PredictResponse(BaseModel):
    disc_scores: dict
    disc_breakdown_scores: list
    disc_scores_high_low: dict
    disc_break_down_scores_high_low: dict
    personalized_feedback: str
    feedback_by_each_traits: list
    strengths: str
    weaknesses: str

@app.middleware("http")
async def verify_secret_token(request: Request, call_next):
    secret_token = request.headers.get("disc-secret-token")
    ##################################################
    path = request.url.path

    # Exclude Swagger documentation routes from authentication
    if path.startswith("/docs") or path.startswith("/redoc") or path.startswith("/openapi.json"):
        response = await call_next(request)
    else:
        secret_token = request.headers.get("disc-secret-token")
        print("checking token: ", secret_token)
        if secret_token != SECRET_TOKEN:
            raise HTTPException(status_code=401, detail="Forbidden, Token not found")
        response = await call_next(request)
    ##################################################

    return response



@app.get("/")
async def read_root():
    df = pd.read_csv("disc_questions.csv")

    num_records_to_select = 2

    # Create an empty DataFrame to store the selected records
    selected_records = pd.DataFrame()

    # Group the DataFrame by 'Category' and randomly select 3 records from each group
    for category, group in df.groupby('trait'):
        if len(group) >= num_records_to_select:
            selected_group = group.sample(n=num_records_to_select, random_state=random.seed())
            selected_records = selected_records._append(selected_group)

    # Display the randomly selected records
    selected_records = selected_records[['question', 'question_code', 'type', 'trait', 'score_type']]
    selected_records = selected_records.sort_values(by='type')
    questions = selected_records[['question_code','question']].reset_index(drop=True).to_dict('records')
    return questions

###############################################################################
@app.post("/disc_scores", response_model=PredictResponse)
async def predict_score(data: PredictReq):
    uri = "mongodb+srv://readonly_user_dev:kmJMIj0P8RHHRAxj@student.qwtbgls.mongodb.net/?retryWrites=true&w=majority"
    # Replace with your MongoDB connection string
    client = MongoClient(uri)

    database = client['DEV_STUDENT']

    # database = client['dev_student']
    collection = database['students']

    cursor = collection.find({"preference": {"$exists": True}}, {'first_name': 1, 'last_name': 1,
                                                                 'basic_info': 1, 'education_records': 1, 'courses': 1,
                                                                 'skills': 1, 'subjects': 1, 'tools': 1,
                                                                 'work_experiences': 1, 'preference': 1})

    students_df = pd.DataFrame(cursor)
    students_df['job_positions'] = students_df['preference'].apply(lambda x: x['job_positions'])
    students_df['job_positions'] = students_df['job_positions'].apply(list_to_string)

    students_df['industry'] = students_df['preference'].apply(lambda x: x['industry'])
    students_df['industry'] = students_df['industry'].apply(list_to_string)
    students_df['_id'] = students_df['_id'].astype(str)
    students_df['job_positions'] = students_df['job_positions'].astype(str)
    students_df['industry'] = students_df['industry'].astype(str)

    student_id = jsonable_encoder(data)['student_id']
    specific_student = students_df[students_df['_id'] == student_id]
    first_name = "".join(str(i) for i in list(specific_student["first_name"]))
    last_name = "".join(str(i) for i in list(specific_student["last_name"]))
    job_pos = "".join(str(i) for i in list(specific_student["job_positions"]))
    industry = "".join(str(i) for i in list(specific_student["industry"]))
    print("#####", first_name)
    print("#####", last_name)
    print("#####", job_pos)
    print("#####", industry)

    df = pd.read_csv("disc_questions.csv")
    sample_payload = jsonable_encoder(data)['disc_payload']
    # filtered_df = df.loc[df['question_code'].isin(list(sample_payload.keys()))]
    # filtered_df = filtered_df.drop(columns=['Unnamed: 0'])
    # filtered_df['scores'] = filtered_df['question_code'].map(sample_payload)
    # filtered_df['scores'] = filtered_df['scores'] * 16.67
    # filtered_df['scores'] = filtered_df['scores'].apply(lambda x: round(x, 3))
    #
    # remapvalues_for_minus = {20: 100, 40: 80, 80: 40, 100: 20, 60: 60}
    # filtered_df["updated_scores_for_minus"] = filtered_df[filtered_df['score_type'] == 'minus']['scores'].replace(
    #     remapvalues_for_minus)
    #
    # df_minus = filtered_df[
    #     ["question", "type", "trait", "score_type", "question_code", "updated_scores_for_minus"]].dropna()
    # filtered_df = filtered_df[filtered_df['score_type'] != 'minus'].drop(columns=['updated_scores_for_minus'])
    #
    # df_minus = df_minus.rename(columns={'updated_scores_for_minus': 'scores'})
    # result = pd.concat([filtered_df, df_minus])
    # result['scores'] = result['scores'].apply(lambda x: round(x, 3))
    #
    # result_break_downs = result.groupby(['type', 'trait'])['scores'].sum()
    # result_break_downs = result_break_downs.reset_index()
    # break_down = result_break_downs.set_index(["type","trait"])["scores"].to_dict()
    # break_down_for_response = result_break_downs.to_dict('records')
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #
    # result_disc = result.groupby('type')['scores'].sum()
    #
    # print("231023@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # result_disc = result_disc.reset_index()
    # disc_scores = result_disc.set_index("type")["scores"].to_dict()
    # print(type(disc_scores))
    #
    # ##########################DISC HIGH LOW
    #
    # import numpy as np
    # conditions1 = [
    #     (result_disc['scores'] < 40),
    #     (result_disc['scores'] >= 40) & (result_disc['scores'] < 60),
    #     (result_disc['scores'] >= 60) & (result_disc['scores'] <= 100)
    # ]
    # values1 = ['Low', 'Neutral', 'High']
    #
    # result_disc['highlow_disc'] = np.select(conditions1, values1)
    #
    # highlow_values = result_disc[['type', 'highlow_disc']].set_index("type")["highlow_disc"]
    # highlow_values = highlow_values.to_dict()
    #
    # conditions2 = [
    #     (result_break_downs['scores'] < 13.34),
    #     (result_break_downs['scores'] >= 13.34) & (result_break_downs['scores'] < 20),
    #     (result_break_downs['scores'] >= 20) & (result_break_downs['scores'] <= 33.34)
    # ]
    # values2 = ['Low', 'Neutral', 'High']
    #
    # result_break_downs['highlow_break_downs'] = np.select(conditions2, values2)
    #
    # highlow_values_break_downs = result_break_downs[['trait', 'highlow_break_downs']].set_index("trait")[
    #     "highlow_break_downs"]
    # highlow_values_break_downs = highlow_values_break_downs.to_dict()

    filtered_df = df.loc[df['question_code'].isin(list(sample_payload.keys()))]
    filtered_df = filtered_df.drop(columns=['Unnamed: 0'])
    filtered_df['scores'] = filtered_df['question_code'].map(sample_payload)
    filtered_df['scores'] = filtered_df['scores'] * 100
    remapvalues_for_minus = {20: 100, 40: 80, 80: 40, 100: 20, 60: 60}
    filtered_df["updated_scores_for_minus"] = filtered_df[filtered_df['score_type'] == 'minus']['scores'].replace(
        remapvalues_for_minus)

    df_minus = filtered_df[
        ["question", "type", "trait", "score_type", "question_code", "updated_scores_for_minus"]].dropna()
    filtered_df = filtered_df[filtered_df['score_type'] != 'minus'].drop(columns=['updated_scores_for_minus'])
    df_minus = df_minus.rename(columns={'updated_scores_for_minus': 'scores'})
    result = pd.concat([filtered_df, df_minus])
    result['scores'] = result['scores'].apply(lambda x: round(x, 3))
    result['average_scores'] = result.groupby('trait')['scores'].transform('mean')
    # result.to_csv("disc_result_1103.csv")

    unique_rows = result[['type',"trait","average_scores"]].drop_duplicates()
    unique_rows['average_scores'] = unique_rows['average_scores']/3
    unique_rows = unique_rows [['type',"trait","average_scores"]]
    unique_rows = unique_rows.rename(columns={'average_scores': 'scores'})
    unique_rows_dict = unique_rows.to_dict(orient='records')

    result_dict = result.groupby('trait')['average_scores'].unique().apply(lambda x: x[0]).to_dict()

    conditions1 = [
        (result['average_scores'] < 40),
        (result['average_scores'] >= 40) & (result['average_scores'] < 60),
        (result['average_scores'] >= 60) & (result['average_scores'] <= 100)
    ]
    values1 = ['Low', 'Neutral', 'High']

    result['highlow_tags'] = np.select(conditions1, values1)

    highlow_values = result[['trait', 'highlow_tags']].set_index("trait")["highlow_tags"]
    highlow_values_traits = highlow_values.to_dict()

    result_dict = {k: v for k, v in map(lambda kv: (kv[0], kv[1] / 3), result_dict.items())}
    print("@@@@@@@@@@@@@@@@@11111",result_dict)
    result['average_scores_type'] = result.groupby('type')['scores'].transform('mean')
    result_dict_type = result.groupby('type')['average_scores_type'].unique().apply(lambda x: x[0]).to_dict()

    conditions1 = [
        (result['average_scores_type'] < 40),
        (result['average_scores_type'] >= 40) & (result['average_scores_type'] < 60),
        (result['average_scores_type'] >= 60) & (result['average_scores_type'] <= 100)
    ]
    values1 = ['Low', 'Neutral', 'High']

    result['highlow_tags_type'] = np.select(conditions1, values1)

    highlow_values_type = result[['type', 'highlow_tags_type']].set_index("type")["highlow_tags_type"]
    highlow_values_type = highlow_values_type.to_dict()


    breakdown_feedbacks = []

    for i in result_dict.items():
        breakdown_feedbacks.append(
            {str(i[0]): re.sub(r'\s+', ' ', re.sub(r'\{[^{}]*\}|\([^()]*\)', ' ', str(break_down_feedback(str(i)))))})

    prompt1901 = str("""Generate Overall description to the following Student in  3 lines based on following data to improve their overall personality.
        Start the response with You. Do not add sincerely, best regards at the end of response.""" + "Below are the student details\n\n" +

                 """Student Name: """ + first_name + "\n\n" + """DISC Personality Score : """ + str(result_dict_type) + "\n\n" + """Desired Jobs: """ + job_pos + "\n" + """Desired Industrries: """ + industry)
    print("##############", prompt1901)

    prompt26 = str("""list out the strengths to the student with a short explanation in 3 lines based on below DISC numerical scores and follow rules.
        
        Follow the below rules strictly while generating response.

        1. Please note the following  every score data is out of 33.33
        2. Strictly do not include any numerical score value in response.
        3. Below each score is out of 33.33. Strictly start the response with You. Do not start the response as Below are the student's details""" + "Below are the student's score details\n\n" +

                    """DISC Personality Score : """ + str(
        result_dict) + "\n\n" + """Desired Jobs: """ + job_pos + "\n" + """Desired Industrries: """ + industry + "\n\n Strictly start the response as You.")
    print("##############", prompt26)

    headers = {
        'Authorization': 'Bearer ' + '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',
        'Content-Type': 'application/json',
    }

    json_data = {
        'model': 'gpt-3.5-turbo-16k',
        'temperature': 0.2,
        'messages': [
            {
                'role': 'user',
                'content': prompt26
            }
        ],
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data)

    # Process the response
    if response.status_code == 200:
        res = response.json()
        resp26 = res['choices'][0]['message']['content']
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)

    headers = {
        'Authorization': 'Bearer ' + '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',
        'Content-Type': 'application/json',
    }

    json_data = {
        'model': 'gpt-3.5-turbo-16k',
        'temperature': 0.2,
        'messages': [
            {
                'role': 'user',
                'content': prompt1901
            }
        ],
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data)

    # Process the response
    if response.status_code == 200:
        res = response.json()
        resp1901 = res['choices'][0]['message']['content']
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)


    prompt27 = str("""list out the weeknesses to the student in 3 lines with a short explanation based on below DISC numerical scores breakdown by each personality traits and follow rules.
        Follow the below rules strictly while generating response.
        
        1. Please note the following  every score data is out of 33.33
        2. Strictly do not include any numerical value in response.
        3. Strictly start the response with You. Do not start the response as Below are the student's details """ + "Below are the student's score details\n\n" +

                    """DISC Personality Score : """ + str(
        result_dict) + "\n\n" + """Desired Jobs: """ + job_pos + "\n" + """Desired Industrries: """ + industry + "\n\n Strictly start the response with You.")
    print("##############", prompt27)

    headers = {
        'Authorization': 'Bearer ' + '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',
        'Content-Type': 'application/json',
    }

    json_data = {
        'model': 'gpt-3.5-turbo-16k',
        'temperature': 0.2,
        'messages': [
            {
                'role': 'user',
                'content': prompt27
            }
        ],
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data)

    # Process the response
    if response.status_code == 200:
        res = response.json()
        resp27 = res['choices'][0]['message']['content']
        # return resp27
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)


    return PredictResponse(
                           disc_scores=result_dict_type,
                           disc_breakdown_scores=unique_rows_dict,
                           disc_scores_high_low=highlow_values_type,
                           disc_break_down_scores_high_low=highlow_values_traits,
                           personalized_feedback=resp1901,
                           feedback_by_each_traits=breakdown_feedbacks,
                           strengths=resp26,
                           weaknesses=resp27
        )