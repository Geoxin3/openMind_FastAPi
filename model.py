from transformers import pipeline, AutoTokenizer

#pre trained model from hugging face
emotion_detector = pipeline(
    "text-classification",
    model="borisn70/bert-43-multilabel-emotion-detection",
    top_k=None)#use None instead of True for returning all emotions

tokenzier = AutoTokenizer.from_pretrained(
    "borisn70/bert-43-multilabel-emotion-detection")

#method for claculating threshold for adjusting dynamically
def calc_threshold(text):
    #encoding tokens
    tokens = tokenzier.encode(text)
    #calculating the length of tokens
    num_tokens = len(tokens)
    #print(num_tokens)
    if num_tokens <= 10:
        return 0.01
    elif num_tokens <= 50:
        return 0.001
    elif num_tokens <= 100:
        return 0.0001
    else:
        return 0.00001

#Detect all emotions in the text and return confidence scores.
def detect_emotions(text):
    #result form model
    emotions = emotion_detector(text)[0]
    threshold = calc_threshold(text)
    #preparing the json format 
    emotion_summmary = {item['label']: item['score'] for item in emotions if item['score'] >= threshold}
    return emotion_summmary
