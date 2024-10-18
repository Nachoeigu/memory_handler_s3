import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

import boto3
import logging
from pydantic import BaseModel, field_validator
from typing import List, Literal, Optional
import re
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
import uuid

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Message(BaseModel):
    role: Literal['system', 'human', 'ai']
    message: str
    etl_time: str

    @field_validator('etl_time')
    def validate_etl_time(cls, v):
        if not re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$', v):
            raise ValueError(f"Invalid datetime format: {v}. Expected format is 'YYYY-MM-DDTHH:MM:SS.sssZ'.")
        return v

class ChatHistory(BaseModel):
    data: List[Message]

    @field_validator('data')
    def ensure_non_empty(cls, v):
        if len(v) == 0:
            raise ValueError('Chat history must contain at least one message')
        return v


class MemoryHandlerS3:
    def __init__(self):
        """
        Initialize the MemoryHandlerS3 class.
        """
        self.s3 = boto3.client('s3')
        self.bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        
        if not self.bucket_name:
            raise ValueError("Bucket name not found. Please provide a bucket name or set the 'AWS_S3_BUCKET_NAME' environment variable.")
        
    def __adding_new_data_in_previous_chat_history(self, existing_data: ChatHistory, new_data: ChatHistory):            
            full_data = existing_data.data
            for msg in new_data.data:
                if msg not in existing_data.data:
                    full_data.append(msg)

            full_data = ChatHistory(data=full_data)

            return full_data

    def upload_chat_history(self, user_id: str, new_data: ChatHistory) -> bool:
        """
        Upload chat history for a user to S3. If the chat history already exists,
        append the new entries to the existing chat history, preserving the 'etl_time'
        for existing messages and adding it to new ones.

        Args:
            user_id (str): The ID of the user whose chat history is being uploaded.
            new_data (ChatHistory): The new chat history data to append (validated by Pydantic).

        Returns:
            bool: True if the process worked successfully, False otherwise.
        """
        s3_key = f'{user_id}/chat.json'
        
        existing_data = self.retrieve_chat_history(user_id)
        if existing_data is None:
            self.s3.put_object(Bucket=self.bucket_name, Key=s3_key, Body=new_data.json())
            return True
        else:
            full_data = self.__adding_new_data_in_previous_chat_history(existing_data=existing_data, new_data=new_data)
            self.s3.put_object(Bucket=self.bucket_name, Key=s3_key, Body=full_data.json())
            return True

    def retrieve_chat_history(self, user_id) -> Optional[str]:
        """
        Retrieve chat history for a user from S3.

        Args:
            user_id (str): The ID of the user whose chat history is being retrieved.

        Returns:
            str: The contents of the chat history, or None if an error occurred.
        """
        s3_key = f'{user_id}/chat.json'
        
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            data = response['Body'].read().decode('utf-8')
            return ChatHistory(**json.loads(data))
        except Exception as e:
            return None

    def delete_chat_history(self, user_id: str) -> bool:
        """
        Delete a user's chat history from S3.

        Args:
            user_id (str): The ID of the user whose chat history is being deleted.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        s3_key = f'{user_id}/chat.json'
        
        response = self.s3.delete_object(Bucket=self.bucket_name, Key=s3_key)
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            return False
        else:
            return True
        
    def convert_memory_in_langchain_obj(self, chat_history: ChatHistory):
        return [SystemMessage(content=message.message, etl_time=message.etl_time) if message.role == 'system' 
          else HumanMessage(content=message.message, id=str(uuid.uuid4()), etl_time=message.etl_time) if message.role == 'human' 
          else AIMessage(content=message.message, id=str(uuid.uuid4()), etl_time=message.etl_time) 
          for message in chat_history.data]
    
    def convert_langchain_obj_in_memory(self, chat_history: List[AnyMessage]):
        return {'data': [{'role': msg.type, 'message': msg.content, 'etl_time': msg.etl_time} for msg in chat_history]}

