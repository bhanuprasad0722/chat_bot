import os
import ssl
import random
import nltk #used for tokenization and text processing
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# SSL fix for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define chatbot intents
intents = [
    {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "How are you?", "What's up?", "Good morning"],
     "responses": ["Hello! How can I assist you today?", "Hey there! Need any help?", "Hi! What can I do for you?", "I'm doing great, thanks for asking! How about you?"]},

    {"tag": "goodbye", "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
     "responses": ["Goodbye! Have a great day!", "See you soon!", "Take care and stay safe!", "Bye! Feel free to chat anytime!"]},

    {"tag": "thanks", "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
     "responses": ["You're very welcome!", "Happy to help!", "Anytime! Let me know if you need anything else."]},

    {"tag": "about", "patterns": ["What can you do?", "Who are you?", "What is your purpose?", "Tell me about yourself"],
     "responses": ["I'm a chatbot created to assist you with various queries!", "I can answer questions, chat with you, and offer guidance.", "My purpose is to make conversations more engaging and helpful!"]},

    {"tag": "jokes", "patterns": ["Tell me a joke", "Make me laugh", "Do you know any jokes?"],
     "responses": ["Why don't programmers like nature? It has too many bugs!", "I told my computer I needed a break, and now it won’t stop sending me vacation ads!", "Why was the JavaScript developer sad? Because he didn’t ‘null’ his problems!"]},

    {"tag": "weather", "patterns": ["What’s the weather like?", "Is it going to rain today?", "Tell me the weather"],
     "responses": ["I can't check live weather, but you can try a weather app!", "I recommend checking a weather website for real-time updates."]},
         {"tag": "math", "patterns": ["What is 2 + 2?", "Solve 5 * 6", "What is 10 squared?"],
     "responses": ["2 + 2 is 4!", "5 * 6 is 30!", "10 squared is 100!"]},

    {"tag": "science", "patterns": ["What is gravity?", "Tell me about space", "How does the sun produce energy?"],
     "responses": ["Gravity is the force that attracts objects toward each other.", "Space is a vast, empty area beyond Earth's atmosphere.", "The sun produces energy through nuclear fusion!"]},

    {"tag": "history", "patterns": ["Who was the first president of the USA?", "Tell me about World War 2", "When was the moon landing?"],
     "responses": ["The first U.S. president was George Washington.", "World War 2 lasted from 1939 to 1945.", "The first moon landing was in 1969, by Apollo 11."]},

    {"tag": "geography", "patterns": ["What is the capital of France?", "Which is the largest ocean?", "Tell me about Mount Everest"],
     "responses": ["The capital of France is Paris.", "The Pacific Ocean is the largest ocean on Earth.", "Mount Everest is the highest mountain, located in Nepal/Tibet."]},

    {"tag": "technology", "patterns": ["What is AI?", "What is machine learning?", "What is the Internet?"],
     "responses": ["AI stands for Artificial Intelligence, allowing machines to mimic human intelligence.", "Machine learning is a branch of AI that enables systems to learn from data.", "The Internet is a global network that connects computers worldwide."]},
        {"tag": "time", "patterns": ["What time is it?", "Tell me the current time"],
     "responses": ["I don't have a clock, but you can check your phone!", "Try asking Google or checking your watch."]},

    {"tag": "date", "patterns": ["What’s today’s date?", "What day is it?"],
     "responses": ["You can check the date on your device!", "Today's date is easily available on your phone."]},

    {"tag": "motivation", "patterns": ["Give me some motivation", "I need inspiration", "Tell me a motivational quote"],
     "responses": ["Believe in yourself! You are capable of great things.", "The only limit to your success is your mindset!", "Success is not final, failure is not fatal—it is the courage to continue that counts."]},

    {"tag": "health", "patterns": ["How can I stay healthy?", "What are some healthy foods?", "Give me fitness tips"],
     "responses": ["Eat balanced meals, exercise regularly, and get enough sleep!", "Healthy foods include fruits, vegetables, nuts, and lean proteins.", "Stay active, drink plenty of water, and maintain a good sleep schedule."]},
    {"tag": "movies", "patterns": ["Recommend a movie", "What’s a good movie to watch?", "Give me a movie suggestion"],
     "responses": ["Try watching 'Inception' if you like sci-fi!", "If you enjoy action, 'John Wick' is a great choice!", "How about a comedy? 'The Office' is hilarious!"]},

    {"tag": "music", "patterns": ["Recommend a song", "What’s a good song?", "Who is your favorite musician?"],
     "responses": ["Try listening to 'Blinding Lights' by The Weeknd!", "If you like pop, check out Taylor Swift!", "For classical music, Beethoven is a legend!"]},

    {"tag": "sports", "patterns": ["What’s the latest football match result?", "Tell me about cricket", "Who won the NBA championship?"],
     "responses": ["I don't have live updates, but you can check ESPN!", "Cricket is a popular sport in countries like India and England.", "The latest NBA champion can be found on sports websites."]},
    {"tag": "coding", "patterns": ["How do I learn Python?", "What is JavaScript used for?", "Explain HTML"],
     "responses": ["You can start learning Python with online courses like Codecademy!", "JavaScript is used for making interactive web pages.", "HTML is the standard language for creating web pages."]},

    {"tag": "cybersecurity", "patterns": ["What is phishing?", "How can I stay safe online?", "Tell me about hacking"],
     "responses": ["Phishing is a type of cyber attack where attackers trick you into giving personal information.", "Stay safe online by using strong passwords and avoiding suspicious links.", "Hacking refers to gaining unauthorized access to systems. Ethical hacking is used for security testing."]},
    {"tag": "random_fact", "patterns": ["Tell me a fact", "Give me a random fact", "Surprise me!"],
     "responses": ["Did you know honey never spoils?", "A day on Venus is longer than a year on Venus!", "Octopuses have three hearts!"]},

    {"tag": "riddles", "patterns": ["Tell me a riddle", "Give me a puzzle", "Challenge me"],
     "responses": ["What has keys but can't open locks? A piano!", "The more you take, the more you leave behind. What is it? Footsteps!", "I speak without a mouth and hear without ears. I have nobody, but I come alive with the wind. What am I? An echo!"]},

    {"tag": "languages", "patterns": ["How do you say hello in Spanish?", "Translate hello to French"],
     "responses": ["Hello in Spanish is 'Hola'!", "In French, hello is 'Bonjour'!", "In German, hello is 'Hallo'!"]}


]
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Prepare training data
tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

pattern_embeddings = model.encode(patterns)


def chatbot_response(input_text):
    input_embeddings = model.encode([input_text])
    similarity = cosine_similarity(input_embeddings,pattern_embeddings)
    best_match = similarity.argmax()
    predicted_tag = tags[best_match]
    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that. Could you clarify?"

# Initialize session state if not present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def main():
    st.title("Chatbot")
    st.write("Welcome! Type your message below and press Enter to chat.")
    
    # Display chat history
    for role, msg in st.session_state["messages"]:
        st.markdown(f"**{role}:** {msg}")
    
    # User input box at the bottom
    user_input = st.text_input("Type your message:", key="user_input", on_change=handle_user_input)
    
    # Clear chat button
    if st.button("Clear Chat"):
        clear_chat()
        #st.experimental_set_query_params(dummy=str(random.random()))  # Forces UI refresh

def clear_chat():
    """Clears the chat history."""
    st.session_state.messages = []  # Reset the messages list

def handle_user_input():
    user_text = st.session_state["user_input"].strip()
    if user_text:
        response = chatbot_response(user_text)
        
        # Append user input and chatbot response to session state
        st.session_state["messages"].append(("You", user_text))
        st.session_state["messages"].append(("Chatbot", response))

if __name__ == '__main__':
    main()