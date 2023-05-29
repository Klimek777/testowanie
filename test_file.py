import requests
import unittest
from unittest.mock import MagicMock
import pytest
#1. test if the endpoint / return results for 'business' value
def test_endpoint():
    response = requests.get("http://127.0.0.1:5000/")
    data = response.json()

    assert response.status_code == 200
    assert data["Subreddit"] == "business"
    assert "Negative" in data
    assert "Neutral" in data
    assert "Positive" in data


#2.test if custom query will retrun right results for this query
def test_custom_query():
    query = "python"
    response = requests.get(f"http://127.0.0.1:5000/?query={query}")
    data = response.json()

    assert response.status_code == 200
    assert data["Subreddit"] == query
    assert "Negative" in data
    assert "Neutral" in data
    assert "Positive" in data


#3. test if the values of sentiment analysis are correct and within expected range of values

def test_sentiment_scores():
    response = requests.get("http://127.0.0.1:5000/")
    data = response.json()

    assert response.status_code == 200
    assert isinstance(float(data["Negative"]), float)
    assert isinstance(float(data["Neutral"]), float)
    assert isinstance(float(data["Positive"]), float)
    assert 0 <= float(data["Negative"]) <= 1
    assert 0 <= float(data["Neutral"]) <= 1
    assert 0 <= float(data["Positive"]) <= 1


#4. test if the model is loading in a right way and returns the right data

def test_model_loading():
    response = requests.get("http://127.0.0.1:5000/")
    data = response.json()

    assert response.status_code == 200
    assert "Negative" in data
    assert "Neutral" in data
    assert "Positive" in data
    assert data["Negative"] != "0.0"
    assert data["Neutral"] != "0.0"
    assert data["Positive"] != "0.0"


#5. test if the right data format is being returned 
def test_response_format():
    response = requests.get("http://127.0.0.1:5000/")
    data = response.json()

    assert response.status_code == 200
    assert isinstance(data, dict)
    assert "Subreddit" in data
    assert isinstance(data["Subreddit"], str)
    assert "Negative" in data
    assert "Neutral" in data
    assert "Positive" in data
    assert isinstance(data["Negative"], str)
    assert isinstance(data["Neutral"], str)
    assert isinstance(data["Positive"], str)


#6. test if the received sentiment scores have correct sums for each category.
def test_sentiment_sums():
    response = requests.get("http://127.0.0.1:5000/")
    data = response.json()

    assert response.status_code == 200
    assert isinstance(float(data["Negative"]), float)
    assert isinstance(float(data["Neutral"]), float)
    assert isinstance(float(data["Positive"]), float)
    assert 0 <= float(data["Negative"]) <= 1
    assert 0 <= float(data["Neutral"]) <= 1
    assert 0 <= float(data["Positive"]) <= 1
    assert (
        round(float(data["Negative"]) + float(data["Neutral"]) + float(data["Positive"]), 2)
        == 1.0
    )

##atrapy


#7.mock the reddit api 
@pytest.mark.parametrize("client_id, client_secret, user_agent", [("id_1", "secret_1", "agent_1")])
def test_mocked_subreddit(client_id, client_secret, user_agent):
    class RedditMock:
        def __init__(self, client_id, client_secret, user_agent):
            pass

        def subreddit(self, name):
            subreddit_mock = MagicMock()
            subreddit_mock.top.return_value = [
                MagicMock(title="Post 1"),
                MagicMock(title="Post 2"),
                MagicMock(title="Post 3")
            ]
            return subreddit_mock

    reddit_mock = RedditMock(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    subreddit = reddit_mock.subreddit("python")

    assert len(subreddit.top()) == 3
    assert subreddit.top()[0].title == "Post 1"
    assert subreddit.top()[1].title == "Post 2"
    assert subreddit.top()[2].title == "Post 3"

#8. softmax mock

def softmax_mock(scores, axis):
    softmax_scores_mock = MagicMock()
    softmax_scores_mock.sum.return_value = [0.1, 0.2, 0.7]
    return softmax_scores_mock

softmax = softmax_mock
def test_softmax_mock():
    scores = [1.0, 2.0, 3.0]
    axis = 1
    expected_output = [0.1, 0.2, 0.7]

    output = softmax(scores, axis)

    assert output.sum() == expected_output


#9.AutoModel mock
class AutoModelForSequenceClassificationMock:
    def __init__(self, pretrained_model):
        pass

    def __call__(self, *args, **kwargs):
        outputs_mock = MagicMock()
        outputs_mock[0].detach().numpy.return_value = [[0.2, 0.3, 0.5]]
        return outputs_mock

model_mock = AutoModelForSequenceClassificationMock(pretrained_model="")

def test_model_output():

    inputs = "Przykladowy tekst do tego testu"
    expected_output = [[0.2, 0.3, 0.5]]
    output = model_mock(inputs)

    assert output[0].detach().numpy() == expected_output


##10. autotokenizer mock

class AutoTokenizerMock:
    def __init__(self, pretrained_model):
        pass

    def __call__(self, *args, **kwargs):
        encoded_reddits_mock = MagicMock()
        encoded_reddits_mock.return_tensors.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1]
        }
        return encoded_reddits_mock

tokenizer_mock = AutoTokenizerMock(pretrained_model="")


def test_tokenizer_output():
    input_text = "Przykładowy tekst do tego testu"
    expected_output = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1]
    }
    
    tokenizer = AutoTokenizerMock(pretrained_model="")
    output = tokenizer(input_text)
    
    assert output.return_tensors.return_value == expected_output



