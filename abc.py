#!/usr/bin/env python
# coding: utf-8

# # 🤖 Capstone Project: AgentRed - AI-Powered Debate System using Generative AI
# 
# ## 📌 Introduction
# 
# As part of the **Kaggle 5-Day Generative AI Intensive Course**, this capstone project explores an innovative application of GenAI: simulating debates between opposing public narratives using AI agents. The system not only gathers and analyzes online opinions but also engages in a logic-based debate to determine which side presents a more compelling argument.
# 
# ### 🎯 Project Objectives
# 
# The key features of this AI debate system are:
# 
# 1. **Dominant Sentiments Detection**  
#    Extract and analyze the prevailing public sentiments around a specific topic or event from online platforms.
# 
# 2. **Popular Opinion Mining**  
#    Identify the most influential or frequently expressed opinions to represent both sides of a debate.
# 
# 3. **AI Debate Simulation**  
#    Generate a debate between two AI agents representing each dominant narrative. A third AI agent acts as a **Judge**, evaluating the arguments and delivering a **verdict** on which side won based on logic, consistency, and supporting evidence.
# 
# ---
# 
# ### 💡 Core Concepts Used
# 
# - **Prompt Engineering**: Carefully crafted prompts guide the behavior of each AI agent to ensure coherent and context-aware responses.
# - **Agents**: Specialized AI components with designated roles: data retriever, debaters, and judge.
# - **Chat Persistence (Memory)**: Maintains contextual understanding across multiple exchanges, simulating a more natural and informed conversation.
# 
# ---
# 
# ### 🔁 System Flow
# 
# ```text
# User Input (e.g., "Mike Tyson vs Logan Paul")
#         ↓
#      [Agent]
#         ↓ (Searches & Retrieves using Reddit PRAW API)
#  ┌──────────────────────────────┐
#  │     Dominant Narratives      │ ←←←←←←←←←←←←←←←←←←←←←←←←←┐
#  └──────────────────────────────┘                          │
#  ┌──────────────────────────────┐                          │
#  │      Sentiment Analysis      │ →→→→→→→→→→→→→→→→→→→→→→→→→┘
#  └──────────────────────────────┘
#         ↓
#   Opinion #1 vs Opinion #2
#         ↓
# ┌──────────────┐     ┌──────────────┐
# │    Chad      │     │    Wojak     │
# │ (Debater 1)  │     │ (Debater 2)  │
# └──────────────┘     └──────────────┘
#         ↓                  ↓
#          ─────→ Judge Agent ←─────
#                       ↓
#                   Verdict (Who won?)
# 

# ### Loading Environment Variables
# 
# This cell loads sensitive data like API keys from a `.env` file using the `dotenv` library. The keys are accessed using the `os` module to retrieve the `GOOGLE_API_KEY`, `REDDIT_CLIENT_SECRET`, and `REDDIT_CLIENT_ID` for use in interacting with Google and Reddit APIs.
# 

# In[1]:


from dotenv import load_dotenv
import os
import praw
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import Image, display, Markdown


# In[2]:


load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
PRAW_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
PRAW_CLIENT = os.getenv("REDDIT_CLIENT_ID")


# ### Let's Set Up Our LLM!
# 
# In this cell, we set up the `ChatGoogleGenerativeAI` with the model `gemini-2.0-flash-lite`. 

# In[3]:


gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
response = gemini.invoke("Hello World?")
response.content


# ### Setting Up Reddit API Client
# 
# Here, we initialize the `praw.Reddit` client to interact with Reddit. We pass in the necessary credentials like `client_id`, `client_secret`, and `user_agent` to authenticate and connect to Reddit's API. This sets us up to retrieve data from Reddit or perform actions like posting or commenting.
# 

# In[4]:


reddit = praw.Reddit(
    client_id=PRAW_CLIENT,
    client_secret=PRAW_SECRET,
    user_agent="my user agent",
)


# 
# ### Fetching Hot Posts from the Kaggle Subreddit
# 
# In this cell, we retrieve the top hot post from the "Kaggle" subreddit using the `reddit.subreddit("Kaggle").hot()` method. We limit the results to 4 post and print the post's title and content (`selftext`). This gives us a quick look at the current hot discussions on Kaggle.

# In[5]:


for submission in reddit.subreddit("Kaggle").hot(limit=4):
    print(submission.title)
    print("---------------")
    print(submission.selftext)
    


# ### Creating a Function to Retrieve Reddit Submissions
# 
# We will define a tool using the `langchain` library to retrieve Reddit submissions based on a given topic. The function `retrieve_submissions` allows us to specify:
# - `topic`: The keyword or phrase to search for.
# - `k`: The number of submissions to retrieve.
# - `subreddit_name`: The subreddit to search within (default is "all").
# - `sort`: The sorting method (either 'relevance' or 'top').
# 
# The function queries Reddit's API, fetches the top `k` submissions related to the topic, and returns a list of dictionaries containing each submission's title, content, and score. This tool helps automate the process of fetching and processing relevant Reddit posts.
# 

# In[6]:


from langchain.tools import tool
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition 
from langgraph.prebuilt import ToolNode


# In[7]:


@tool
def retrieve_submissions(
    topic: str,
    k: int,
    subreddit_name: str = "all",
    sort: str = "relevance"
) -> list:
    """
    Retrieves the top k Reddit submissions related to a given topic from a subreddit.

    Args:
        topic (str): The topic to search for in Reddit submissions.
        k (int): The number of submissions to retrieve.
        subreddit_name (str): The subreddit to search in (default is 'all').
        sort (str): The sort order ('relevance' or 'top').

    Returns:
        list: A list of dictionaries with 'title', 'content', and 'score' of each submission.
    """
    subreddit = reddit.subreddit(subreddit_name)
    submissions = subreddit.search(topic, limit=k, sort=sort)

    results = []
    for submission in submissions:
        results.append({
            'title': submission.title,
            'content': submission.selftext,
            'score': submission.score
        })
    return results


# ### Binding the Tool to the LLM
# 
# We bind the `retrieve_submissions` tool to the `gemini` model. By doing this, we enable the model to use the `retrieve_submissions` function as part of its workflow, allowing it to fetch relevant Reddit posts based on a query. The `gemini.bind_tools(tools)` method connects the tool with the LLM, making it available for use during the model’s execution.
# 

# In[8]:


tools = [retrieve_submissions]
gemini = gemini.bind_tools(tools)


# ## Setting up the retriever agent
# 
# This code sets up a system where an AI agent analyzes Reddit discussions to find two strong, opposing viewpoints on a specific topic. The purpose is to extract contrasting opinions that could lead to a meaningful debate.
# 
# - A detailed prompt is defined that instructs the AI on how to approach the task. It includes how to retrieve Reddit posts, what kind of content to focus on (highly upvoted or frequently repeated sentiments), and how to determine when two clear, opposing opinions are identified.
# 
# - The AI is expected to use a tool called `retrieve_submissions` to fetch Reddit posts based on parameters like topic, subreddit, number of posts (`k`), and sort order. It starts with `k = 5` and increases by 5 until it finds suitable opposing views or reaches a limit of 50.
# 
# - The reasoning function (`reasoner`) combines the initial system instructions with the current conversation state and uses the Gemini model to generate a response.
# 
# - A `StateGraph` is used to manage the flow of operations:
#   - It starts with the `reasoner` node.
#   - If the AI wants to use a tool (like calling the Reddit API), it transitions to the `tools` node.
#   - After using the tool, it returns to the `reasoner` to continue processing.
#   - This loop continues until no more tool use is required, and the task completes.
# 
# - Finally, the graph structure is compiled and displayed visually, showing how the logic flows between the different parts of the system.
# 

# In[9]:


agent_prompt = """
Your task is to analyze Reddit discussions on a given topic and uncover two **extreme and opposing opinions** that could spark a meaningful debate.

Use the tool `retrieve_submissions(topic: str, k: int, subreddit_name: str = "all", sort: str = "relevance")` to retrieve recent submissions. Begin with `k = 5`, `subreddit_name = "all"`, and `sort = "top"` to prioritize influential posts.

From each retrieved submission (`title`, `content`, and `score`), identify two strongly **contrasting** perspectives. Each should meet at least one of the following criteria:

- Frequently repeated sentiment across multiple posts
- Submissions with high upvotes (score)

If two such opposing viewpoints aren't clearly found, increment `k` by 5 and call the tool again. Repeat this process until you find two distinctly opposing and well-supported opinions or until `k = 50`.

Once found, distill the **Debate Axis** — the core question or conflict between the two positions — and clearly outline the rationale behind each opinion.

🎯 **Output Format**:
Return your findings in the following structure:

---
🔍 **Topic Analyzed**: <insert topic here>  
📊 **Final k Value Used**: <insert k>  
⚔️ **Debate Axis**: <Describe the central issue of disagreement between the two sides>

🟥 **Opinion 1: [Clear title for extreme viewpoint #1]**  
**Position Summary**: <Concise, debate-ready thesis of this perspective>  
**Key Arguments**:
1. Argument 1: <Reason or logic backing the viewpoint>
2. Argument 2: <Additional rationale, thematic or emotional>
**Supporting Examples**:
- "<Title or excerpt from submission>" (score: X)
- "<Title or excerpt from another submission>" (score: X)

🟩 **Opinion 2: [Clear title for extreme viewpoint #2]**  
**Position Summary**: <Contrasting thesis from Opinion 1>  
**Key Arguments**:
1. Argument 1: <Reason or logic backing the viewpoint>
2. Argument 2: <Additional rationale, thematic or emotional>
**Supporting Examples**:
- "<Title or excerpt from submission>" (score: X)
- "<Title or excerpt from another submission>" (score: X)

🧠 Notes:
- Avoid duplicates, spam, and low-effort content
- Focus on well-written, thought-provoking submissions
- Emphasize **why** people hold each position, not just **what** they believe
- If no strong polarity is found by k = 50, summarize the dominant viewpoint instead
📌 Goal: Extract well-structured, opposing Reddit viewpoints that can serve as a foundation for meaningful **debate**, not just casual agreement.
"""

sys_msg = SystemMessage(content=agent_prompt)
def reasoner(state: MessagesState):
   return {"messages": [gemini.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools)) # for the tools

# Add edges
builder.add_edge(START, "reasoner")
builder.add_conditional_edges(
    "reasoner",
    # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
    # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "reasoner")
react_graph = builder.compile()

# Display the graph
# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))


# 
# 
# now we will define a function called `extract_opinions`, which takes a user message as input. This message contains the topic to be analyzed (e.g., "Mike Tyson vs Logan Paul").
# 
# - Inside the function:
#   - The input message is wrapped into a list to fit the expected input format for the reasoning system.
#   - The `react_graph` (previously defined graph that manages the logic between reasoning and tool usage) is invoked with this message.
#   - The output is stored in `messages`, and the final AI-generated message (usually the last item) is also extracted separately.
# 
# - The function returns both the full list of messages and the final output message.
# 
# - After calling `extract_opinions` with a sample topic, the final message content (which contains the debate analysis) is displayed using `Markdown`, rendering the structured output in a readable format.

# In[10]:


def extract_opinions(topic):
    message = HumanMessage(content=topic)
    messages = [message]
    messages = react_graph.invoke({"messages": messages})
    return messages, messages['messages'][-1]


# In[45]:


topic = "will AI replace Software engineers?"
messages, final_output = extract_opinions(topic)


# In[56]:


display(Markdown(final_output.content))


# The function below extracts key components from a structured markdown debate transcript, including the debate axis, titles, summaries, and up to two arguments for each opinion. It uses regular expressions to parse the markdown sections based on recognizable headings and markers, returning the extracted data in a clean, structured format suitable for feeding into an AI debate simulation pipeline.

# In[47]:


import re

def parse_debate_output(output_text: str):
    # Extract Debate Axis
    debate_axis_match = re.search(r"⚔️ \*\*Debate Axis\*\*: (.+?)\n", output_text)
    debate_axis = debate_axis_match.group(1).strip() if debate_axis_match else ""

    # Extract Opinion 1
    opinion_1_title = re.search(r"🟥 \*\*Opinion 1: (.+?)\*\*", output_text)
    opinion_1_summary = re.search(r"\*\*Position Summary\*\*: (.+?)\n", output_text)
    opinion_1_args = re.findall(r"\*\*Key Arguments\*\*:\n1\. (.+?)\n2\. (.+?)\n", output_text)

    opinion_1 = {
        "title": opinion_1_title.group(1).strip() if opinion_1_title else "",
        "summary": opinion_1_summary.group(1).strip() if opinion_1_summary else "",
        "arguments": list(opinion_1_args[0]) if opinion_1_args else []
    }

    # Extract Opinion 2
    opinion_2_title = re.search(r"🟩 \*\*Opinion 2: (.+?)\*\*", output_text)
    opinion_2_summary = re.search(r"\*\*Position Summary\*\*: (.+?)\n", output_text[output_text.find("🟩"):])
    opinion_2_args = re.findall(r"\*\*Key Arguments\*\*:\n1\. (.+?)\n2\. (.+?)\n", output_text[output_text.find("🟩"):])

    opinion_2 = {
        "title": opinion_2_title.group(1).strip() if opinion_2_title else "",
        "summary": opinion_2_summary.group(1).strip() if opinion_2_summary else "",
        "arguments": list(opinion_2_args[0]) if opinion_2_args else []
    }

    return debate_axis, opinion_1, opinion_2

debate_axis, opinion_1, opinion_2 = parse_debate_output(final_output.content)


# In[48]:


debate_axis, opinion_1, opinion_2 = parse_debate_output(final_output.content)


# In[49]:


debate_axis


# In[50]:


opinion_1


# In[51]:


opinion_2


# In[52]:


ai_prompt_support = f"""Alright, buckle up, we're diving into this debate!

Debate Axis:
"{debate_axis}"

You're the one reppin' this squad:

Title: {opinion_1['title']}
Summary: 
{opinion_1['summary']}

Key Points (Max 3 brief ones, 'cause who has time for a novel?):
{"\n".join([f"{i+1}. {arg}" for i, arg in enumerate(opinion_1['arguments'][:3])])}

Your mission:
- Drop your points like they’re hot, but make 'em sharp and snappy.
- If they come at you with counterarguments, clap back with 2 sentences max. We ain’t got time for a thesis.
- Keep it cool, confident, and make them see why you’re totally right on this.

And yeah, keep it to 3 quick paragraphs, don’t get carried away like we’re writing a novel here.
"""



# In[53]:


ai_prompt_counter = f"""Alright, here we go, time to roast this take!

Debate Axis:
"{debate_axis}"

You're playing the villain in this scenario:

Title: {opinion_2['title']}
Summary: 
{opinion_2['summary']}

Key Points (Max 3, because brevity is the soul of wit):
{"\n".join([f"{i+1}. {arg}" for i, arg in enumerate(opinion_2['arguments'][:3])])}

Your mission:
- Hit them with your points like it’s a mic drop, but keep it concise.
- If they throw counterpoints, just hit 'em with a 2-sentence comeback and keep it moving.
- Don’t get distracted. Keep that sass and confidence flowing while you tear down their argument.

3 paragraphs max, let’s not overthink this — we’re here to get to the point, not give a TED talk.
"""


# ### AI Debate Simulation with Support and Counter Arguments
# 
# In this code, we simulate a debate where an AI model generates responses for both the support and counter sides of an argument. The debate progresses through multiple rounds, with each side rebutting the other's points. Here's how it works:
# 
# 1. **Model Initialization**:
#    - We initialize the `ChatGoogleGenerativeAI` model with specific settings:
#      - **Model**: `gemini-2.0-flash-lite`
#      - **Temperature**: `0` (This ensures the model gives deterministic outputs with minimal randomness.)
#      - **Max Tokens**: `None` (No token limit for responses.)
#      - **Timeout**: `None` (No timeout set for the model's responses.)
#      - **Max Retries**: `2` (Retries the model's request up to 2 times in case of failure.)
# 
# 2. **Debate Chains**:
#    - **Support Chain**: This chain generates the initial supporting argument for a particular stance. It includes:
#      - **SystemMessage**: The system sets the context and debate prompt using the `ai_prompt_support` template.
#      - **HumanMessage**: A placeholder for the human input, which asks the model to respond to the latest argument and advance the position.
#      - The chain is connected to the `model` for generating AI responses.
#    - **Counter Chain**: Similarly, the counter chain generates opposing arguments. It uses the `ai_prompt_counter` template and follows the same structure as the support chain.
#    
# 3. **Debate Initialization**:
#    - An empty list `debate_history` is initialized to track the debate's progress (i.e., each argument, rebuttal, and counter-rebuttal).
#    
# 4. **First Round - Initial Arguments**:
#    - The first round begins by generating arguments for both sides:
#      - **Support Response**: The model generates an initial supporting argument based on the `support_chain`.
#      - **Counter Response**: The counter-argument is generated using the `counter_chain`.
#    - Both arguments are appended to the `debate_history` for the next round.
# 
# 5. **Subsequent Rounds**:
#    - For each round (5 rounds in total, adjustable):
#      - **Support Rebuttal**: After each counter-argument, the support chain is used to generate a rebuttal.
#      - **Counter Rebuttal**: Similarly, the counter chain is used for a rebuttal to the support rebuttal.
#    - The arguments, rebuttals, and counter-rebuttals are printed after each round, giving a dynamic flow to the debate.
# 

# In[54]:


# Model Initialization
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Create debate chains
support_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=ai_prompt_support),
    MessagesPlaceholder(variable_name="debate_history"),
    HumanMessage(content="Respond to the latest argument and advance your position:")
]) | model

counter_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=ai_prompt_counter),
    MessagesPlaceholder(variable_name="debate_history"),
    HumanMessage(content="Counter the latest argument and strengthen your opposition:")
]) | model



# Initialize debate history
debate_history = []

# First round - Initial arguments
support_response = support_chain.invoke({"debate_history": debate_history})
debate_history.append(AIMessage(content=support_response.content))
print(f"SUPPORT:\n{support_response.content}\n{'='*50}\n")

counter_response = counter_chain.invoke({"debate_history": debate_history})
debate_history.append(AIMessage(content=counter_response.content))
print(f"COUNTER:\n{counter_response.content}\n{'='*50}\n")

# Subsequent rounds
for round_num in range(5):  # Add more rounds as needed
    # Support rebuttal
    support_rebuttal = support_chain.invoke({"debate_history": debate_history})
    debate_history.append(AIMessage(content=support_rebuttal.content))
    display(Markdown(f"SUPPORT REBUTTAL {round_num+1}:\n{support_rebuttal.content}\n{'='*50}\n"))


    
    # Counter rebuttal
    counter_rebuttal = counter_chain.invoke({"debate_history": debate_history})
    debate_history.append(AIMessage(content=counter_rebuttal.content))
    display(Markdown(f"COUNTER REBUTTAL {round_num+1}:\n{counter_rebuttal.content}\n{'='*50}\n"))



# Here’s a single-paragraph markdown explanation for the judgment code block you provided:
# 
# ---
# 
# ### AI Debate Judgement System: Declaring the Chad and Virgin
# 
# This section introduces a judgment mechanism that uses an AI model to evaluate the outcome of a simulated debate. After collecting the full debate history, a dedicated `judgment_chain` prompts the model to analyze both opinions—labeled "SUPPORT" and "COUNTER"—and declare which made the most compelling, logical, and rational case. The AI judge presents its decision using a structured format with clear headings: **Chad** (winner), **Virgin** (loser), and **Reasoning**, along with assessments of **Argument Strength**, **Logical Coherence**, and **Persuasiveness**. This ensures the verdict not only identifies the stronger side but also provides insight into why the winning opinion prevailed, offering a detailed and entertaining conclusion to the AI debate.

# In[58]:


# Create judgment chain
judgment_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
        You're an AI judge evaluating a debate. Based on the debate history provided, determine which opinion won the debate.
        - Skip unnecessary details and directly judge which opinion (either 'SUPPORT' or 'COUNTER') made the most compelling, logical, and rational case.
        - State which opinion won.
        - Provide clear and concise reasoning to explain why one opinion is superior to the other.
        - Format the judgment with headings, including:
            - **Chad**: The opinion that won.
            - **Virgin**: The opinion that lost.
            - **Reasoning**: A brief, logical explanation of why one opinion is superior.
            - **Argument Strength**: Evaluate the strength of each opinion's argument.
            - **Logical Coherence**: Assess how logically consistent each opinion was throughout the debate.
            - **Persuasiveness**: Discuss how persuasive each opinion was in convincing the audience.
    """),
    MessagesPlaceholder(variable_name="debate_history"),
    HumanMessage(content="Review the debate history and determine which opinion won. Provide reasons for your judgment.")
]) | model

# Judgment - Evaluate who is the 'Chad' and who is the 'Virgin'
judgment_response = judgment_chain.invoke({"debate_history": debate_history})

display(Markdown(f"JUDGEMENT:\n{judgment_response.content}\n{'='*50}\n"))


# In[ ]:




