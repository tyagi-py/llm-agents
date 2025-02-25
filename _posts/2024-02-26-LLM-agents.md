---
title: "LLM Agents"
date: 2024-02-22
---


This blog is a summarization of the second lecture of the UC Berkeley LLM Agents course, taught by Mr. Shunyu Yao, Research Scientist @ OpenAI. In this blog we explore what are LLM Agents.

## What is an "agent"?

Before discussing what's an LLM agent, let's talk about what's and agent in general context.

A simple google search defines agent as:

> An agent is a person or entity that acts on behalf of another person or entity.

or

> An "intelligent" system that interacts with some "environment"

![alt text](/assets/images/2024-02-26-LLM-agents-agent.png)

Some examples:

- **Physical environments**: robot, autonomous car, ...
- **Digital environments**: Siri, AlphaGo, Q learning to play Chrome Dinosaur, ...
- **Humans as environments**: chatbot

Now, we need to explain what is *intelligent* and *environment*. But, the problem is it changes over time. 40 years ago a simple chatbot with if else conditions might be considered as intelligent but now a days event ChatGPT sometimes fails to meet the expectations.

Maybe while reading the blog you can do this exercise of thinking "how to define intelligent?"

### LLM Agent

In order to define LLM agents we need to define three concepts or categories.

### Level 1: Text Agent

- Both action and observation are in textual format.
- Examples: ELIZA, LSTM-DQN.

Obviously, that means you can have text agents that does not use LLMs.
![alt text](/assets/images/2024-02-26-LLM-agents-level-1.png)

### Level 2: LLM Agent

- Text agents that use LLM to act
- Examples: SayCan

![alt text](/assets/images/2024-02-26-LLM-agents-level-2.png)

### Level 3: Reasoning Agent

- Text Agents that uses LLM to reason to act
- Examples: ReAct, AutoGPT

![alt text](/assets/images/2024-02-26-LLM-agents-level-3.png)

You might be getting confused between level 2 and 3. No worries this is what we'll be focusing on in the blog.
