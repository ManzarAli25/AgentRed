### 🧠 Reddit Debate Arena: AI Agents Simulating Polarized Discourse

This project presents an experimental system that simulates debates between AI agents using real Reddit data, blending **LangGraph retrieval**, **LangChain reasoning chains**, and **Reddit’s diverse opinions**. It’s a unique intersection of language modeling, argument mining, and conversational simulation — all centered around real-world internet discourse.

At its core is a **Retriever Agent** built with LangGraph. This agent is responsible for discovering controversial Reddit threads. Given a user-defined topic, it retrieves `k` submissions using a function-calling tool powered by the **PRAW API**. If polarized views aren’t found initially, `k` is incrementally increased until two distinct, opposing opinions are detected.

Once polar viewpoints are identified, the agent performs:
- Extraction of the **debate axis** (i.e., the core point of disagreement)
- Summarization of both sides’ stances
- Extraction of **popular arguments** supporting each side

This processed information is then used to build prompts that feed into two **LangChain chains**: the **Support Chain** and the **Counter Chain**. These simulate two sides of a debate. Over the course of **N rounds**, the chains argue with one another, each building on previous responses using memory and prompt context.

After the debate concludes, a third LangChain — the **Judge Chain** — enters. This judge analyzes the debate history and evaluates the **strength and coherence** of the arguments presented by each side. Finally, it humorously assigns the titles:
- 🏆 **Chad** – for the agent with stronger, well-reasoned points
- 🥀 **Virgin** – for the agent with weaker or less convincing arguments

This project explores multi-agent reasoning, real-world text retrieval, debate simulation, and judgment modeling. It’s ideal for anyone interested in language model behavior, Reddit analysis, or agent-based prompting.

> 💡 Whether you're studying AI argumentation or just want to watch bots fight over internet drama, this project has something for you.
