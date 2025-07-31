import os
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor

# Load environment variables from .env file
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Sample book data
books_db = {
    "Atomic Habits": {
        "author": "James Clear",
        "available": True,
        "location": {"floor": 1, "row": 3, "column": 5},
        "tags": ["self-help", "productivity"]
    },
    "Deep Work": {
        "author": "Cal Newport",
        "available": True,
        "location": {"floor": 2, "row": 1, "column": 2},
        "tags": ["focus", "career", "productivity"]
    },
    "Sapiens": {
        "author": "Yuval Noah Harari",
        "available": False,
        "location": {"floor": 1, "row": 2, "column": 4},
        "tags": ["history", "anthropology"]
    }
}

# Sample user data
user_loans = {
    "alekhya": [
        {"title": "Sapiens", "due_date": "2025-07-20"},
        {"title": "Atomic Habits", "due_date": "2025-07-25"}
    ],
    "suresh": [
        {"title": "Deep Work", "due_date": "2025-07-28"}
    ]
}

user_interests = {
    "alekhya": ["history", "self-help"],
    "suresh": ["focus", "career"]
}

# Tool functions
def search_books(title: str) -> str:
    for book_title, book_info in books_db.items():
        if book_title.lower() == title.lower():
            if book_info["available"]:
                loc = book_info["location"]
                return f"📖 '{book_title}' is available at Floor {loc['floor']}, Row {loc['row']}, Column {loc['column']}."
            else:
                return f"❌ '{book_title}' is currently unavailable."
    return f"🔍 No book titled '{title}' found."

def get_recommendations(username: str) -> str:
    interests = user_interests.get(username.lower())
    if not interests:
        return f"🙁 No interests found for '{username}'."
    recs = [f"- {title} by {info['author']}" for title, info in books_db.items()
            if any(tag in interests for tag in info.get("tags", []))]
    return f"📌 Recommendations for {username}:\n" + "\n".join(recs) if recs else "📚 No recommendations."

def calculate_fine(username: str) -> str:
    today = datetime.today().date()
    total_fine = 0
    fines = []
    for loan in user_loans.get(username.lower(), []):
        due = datetime.strptime(loan["due_date"], "%Y-%m-%d").date()
        if today > due:
            days_overdue = (today - due).days
            fine = days_overdue * 5
            total_fine += fine
            fines.append(f"🔻 {loan['title']} overdue by {days_overdue} days → ₹{fine}")
    return "\n".join(fines) + f"\n💰 Total Fine: ₹{total_fine}" if fines else "✅ No overdue books."

def get_due_reminders(username: str) -> str:
    today = datetime.today().date()
    reminders = []
    for loan in user_loans.get(username.lower(), []):
        due = datetime.strptime(loan["due_date"], "%Y-%m-%d").date()
        days_left = (due - today).days
        if 0 <= days_left <= 3:
            reminders.append(f"⏰ {loan['title']} is due in {days_left} day(s) on {due}")
    return "\n".join(reminders) if reminders else "✅ No upcoming due dates."

# Register tools
tools = [
    Tool(name="SearchBooksTool", func=search_books, description="Search for a book by title."),
    Tool(name="GetRecommendations", func=get_recommendations, description="Get recommendations by username."),
    Tool(name="CalculateFine", func=calculate_fine, description="Calculate user fine."),
    Tool(name="GetDueReminders", func=get_due_reminders, description="Upcoming book due dates.")
]

# Load Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro-preview-03-25",
    temperature=0.3,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Agent setup
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "prefix": """
You are LibroGenie, a smart library assistant for college students.
Help with:
- Finding books
- Recommendations
- Calculating fines
- Reminders

Format:
Question: <user question>
Thought: <reasoning>
Action: <tool>
Action Input: <input>
Observation: <result>
Final Answer: <reply to user>
"""
    }
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent.agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    max_execution_time=60
)

# Streamlit UI
st.set_page_config(page_title="📚 LibroGenie", page_icon="📘")
st.title("📚 LibroGenie – Smart Library Assistant")

query = st.text_input("Ask something (e.g., Where is 'Atomic Habits'? What books do I have?)")

if query:
    sanitized_query = query.strip().replace("?", "")
    with st.spinner("Thinking..."):
        try:
            result = agent_executor.invoke({"input": sanitized_query})
            st.success(result["output"])
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
