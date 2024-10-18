# MemoryHandlerS3: Memory Handling for Chatbots powered by LLMs

## Overview

MemoryHandlerS3 is a Python application designed to manage chat histories efficiently for chatbots. By leveraging AWS S3, this application enables the storage, retrieval, and manipulation of user chat interactions remotely and with persistency.
Built with Pydantic for data validation, it ensures the integrity and correctness of the chat data, providing a reliable foundation for chatbot memory management.

## Key Features

- **Data Storage**: Upload chat histories to an AWS S3 bucket, enabling persistent storage of user interactions.
- **Data Retrieval**: Retrieve existing chat histories for users, facilitating continuity in conversations.
- **Data Management**: Append new messages to existing chat histories and delete chat histories as needed.
- **Validation**: Use Pydantic to validate chat messages and ensure they adhere to the correct format.
- **Integration with LangChain**: Convert stored chat histories into LangChain objects for enhanced processing capabilities.

## Components

1. **Message Model**: Represents a single message in the chat history, including the sender's role (`system`, `human`, `ai`), the message content, and the extraction time (`etl_time`).

2. **ChatHistory Model**: A collection of messages, ensuring it contains at least one entry.

3. **MemoryHandlerS3 Class**: Handles interactions with AWS S3, providing methods to upload, retrieve, and delete chat histories. It also incorporates logging for better tracking and debugging.

## Execution

1) Install the dependencies inside `requirements.txt`
2) Create an empty bucket in S3
3) Populate the .env: use .env.template as reference
4) Set the desired values in the agent_config.yaml file
5) Execute the agent.py file
