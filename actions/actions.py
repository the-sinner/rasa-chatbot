# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

import os
from typing import Any, Text, Dict, List
import pandas as pd
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from scipy import spatial  # for calculating vector similarities for search

class HotelAPI(object):

    def __init__(self):
        self.db = pd.read_csv("model/new_delhi_hotels.csv")
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)

    def fetch_hotels(self):
        return self.db.head()

    def format_restaurants(self, df, header=True) -> Text:        
        data = {'hotels': ['Ashok Hotel', 'Ginger Hotel', 'The Lalit']}
        data = {
            'hotels': ['Ashok Hotel', 'Ginger Hotel', 'The Lalit'],
            'ratings': [4.5, 3.2, 4.3],
        }
        df = pd.DataFrame(data)
        return df.to_csv(index=False, header=header)
        # return df.to_csv(index=False, header=header)


class ChatGPT(object):

    def __init__(self):
        self.url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-3.5-turbo"
        self.headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer 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"
        }
        # f"{os.getenv('OPENAI_API_KEY')}"
        self.prompt = "Answer the following question, based on the data shown. " \
            "Answer in a complete sentence and don't say anything else."

    

    # search function
    def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = query_embedding_response["data"][0]["embedding"]
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]




    def num_tokens(text: str, model: str = GPT_MODEL) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))


    def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
    ) -> str:
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        strings, relatednesses = strings_ranked_by_relatedness(query, df)
        # introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
        introduction = self.prompt
        question = f"\n\nQuestion: {query}"
        message = introduction
        for string in strings:
            # next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
            next_review = f'\n\n{string}\n'
            if (
                num_tokens(message + question, model=model)
                > token_budget
            ):
                break
            else:
                message += next_review
        return message + question


    def ask(
        query: str,
        df: pd.DataFrame = df,
        model: str = GPT_MODEL,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
    ) -> str:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        message = query_message(query, df, model=model, token_budget=token_budget)
        if print_message:
            print(message)
        messages = [
            {"role": "system", "content": "You answer questions about Hotels reviews."},
            {"role": "user", "content": message},
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0
        )
        response_message = response["choices"][0]["message"]["content"]
        return response_message


    
    # def ask(self, restaurants, question):
    #     content  = self.prompt + "\n\n" + hotels + "\n\n" + question
    #     body = {
    #         "model":self.model, 
    #         "messages":[{"role": "user", "content": content}]
    #     }
    #     result = requests.post(
    #         url=self.url,
    #         headers=self.headers,
    #         json=body,
    #     )
    #     return result.json()["choices"][0]["message"]["content"]

hotel_api = HotelAPI()
chatGPT = ChatGPT()

class ActionShowHotels(Action):

    def name(self) -> Text:
        return "action_show_hotels"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        hotels = hotel_api.fetch_hotels()
        results = hotel_api.format_hotels(hotels)
        readable = hotel_api.format_hotels(hotels[['Hotels']], header=False)
        dispatcher.utter_message(text=f"Here are some restaurants:\n\n{readable}")

        return [SlotSet("results", results)]


class ActionRestaurantsDetail(Action):
    def name(self) -> Text:
        return "action_hotels_detail"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        previous_results = tracker.get_slot("results")
        question = tracker.latest_message["text"]
        answer = chatGPT.ask(previous_results + "\n\n" + question)
        dispatcher.utter_message(text = answer)