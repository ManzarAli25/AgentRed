from dotenv import load_dotenv
import os
import praw
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition 
from langgraph.prebuilt import ToolNode
import re
import streamlit as st

GOOGLE_KEY = st.secrets["GOOGLE_API_KEY"]
PRAW_SECRET = st.secrets["REDDIT_CLIENT_SECRET"]
PRAW_CLIENT = st.secrets["REDDIT_CLIENT_ID"]



def get_gemini_client():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.75,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key = GOOGLE_KEY
    )

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

    reddit = praw.Reddit(
    client_id=PRAW_CLIENT,
    client_secret=PRAW_SECRET,
    user_agent="my user agent",
    )

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



def get_gemini_with_tools(tools):
    gemini = get_gemini_client()
    return gemini.bind_tools(tools)


def get_reddit_agent():
    tools = [retrieve_submissions]
    gemini = get_gemini_with_tools(tools)
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
    return react_graph


def extract_opinions(topic):
    message = HumanMessage(content=topic)
    messages = [message]
    react_graph = get_reddit_agent()
    messages = react_graph.invoke({"messages": messages})
    return messages, messages['messages'][-1]

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


def create_support_chain(debate_axis, opinion_1):
    model = get_gemini_client()
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
    

    support_chain = ChatPromptTemplate.from_messages([
        SystemMessage(content=ai_prompt_support),
        MessagesPlaceholder(variable_name="debate_history"),
        HumanMessage(content="Respond to the latest argument and advance your position:")
    ]) | model

    return support_chain


def create_counter_chain(debate_axis, opinion_2):
    model = get_gemini_client()
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
    

    counter_chain = ChatPromptTemplate.from_messages([
        SystemMessage(content=ai_prompt_counter),
        MessagesPlaceholder(variable_name="debate_history"),
        HumanMessage(content="Counter the latest argument and strengthen your opposition:")
    ]) | model

    return counter_chain

def run_debate(support_chain, counter_chain, num_rounds=5):
    """
    Function to run the debate with support and counter arguments.

    Args:
        support_chain (Chain): The support side chain.
        counter_chain (Chain): The counter side chain.
        debate_history (list): List to store debate history.
        num_rounds (int): The number of debate rounds.

    Returns:
        None
    """
    debate_history = []
    # First round - Initial arguments
    support_response = support_chain.invoke({"debate_history": debate_history})
    debate_history.append(AIMessage(content=support_response.content))
    print(f"SUPPORT:\n{support_response.content}\n{'='*50}\n")

    counter_response = counter_chain.invoke({"debate_history": debate_history})
    debate_history.append(AIMessage(content=counter_response.content))
    print(f"COUNTER:\n{counter_response.content}\n{'='*50}\n")

    # Subsequent rounds
    for round_num in range(num_rounds):  # Add more rounds as needed
        # Support rebuttal
        support_rebuttal = support_chain.invoke({"debate_history": debate_history})
        debate_history.append(AIMessage(content=support_rebuttal.content))

        # Counter rebuttal
        counter_rebuttal = counter_chain.invoke({"debate_history": debate_history})
        debate_history.append(AIMessage(content=counter_rebuttal.content))


    return debate_history

def get_judgment_chain(model):
    return ChatPromptTemplate.from_messages([
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

def evaluate_debate(debate_history):

    judgment_chain = get_judgment_chain(get_gemini_client())
    judgment_response = judgment_chain.invoke({"debate_history": debate_history})
    return judgment_response.content








# Example topic
# topic = "climate change"

# # Step 1: Extract Opinions from Reddit
# messages, extracted_output = extract_opinions(topic)

# # Step 2: Parse Debate Output to Extract Key Details
# debate_axis, opinion_1, opinion_2 = parse_debate_output(extracted_output.content)

# # Step 3: Create Chains for Both Support and Counter Arguments
# support_chain = create_support_chain(debate_axis, opinion_1)
# counter_chain = create_counter_chain(debate_axis, opinion_2)

# # Step 4: Run the Debate (with 5 rounds)
# debate_history = run_debate(support_chain, counter_chain, num_rounds=5)

# # Step 5: Get the Judgment (Who won the debate?)
# debate_judgment = evaluate_debate(debate_history)

# # Print the Final Judgment
# print(debate_judgment)
