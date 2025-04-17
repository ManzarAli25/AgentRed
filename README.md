# AgentRed
A LangGraph-based Retriever Agent fetches Reddit posts on a topic using the PRAW API until polar opinions are found. It extracts arguments and summaries, then two LangChain agents (Support &amp; Counter) debate for N rounds. A Judge Chain reviews the debate and crowns the Chad and Virgin based on argument strength.
