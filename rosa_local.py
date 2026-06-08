#!/usr/bin/env python3
import os
from langchain_ollama import ChatOllama
from rosa import ROSA

def main():
    print("Initializing Local Google Gemma 4 Model via Host Network...")

    # Fixed base_url for host-networked containers and updated to gemma4
    llm = ChatOllama(
        model="gemma4", 
        temperature=1.0, # Recommended baseline for Gemma 4
        top_p=0.95,
        top_k=64,
        base_url="http://localhost:11434", # Works because of --net=host
        num_predict=2048
    )

    print("Connecting ROSA to ROS 2 ecosystem...")
    agent = ROSA(ros_version=2, llm=llm)

    print("\n🤖 ROSA Agent (Gemma 4) ready! Ask a question about your robot setup.")
    while True:
        try:
            user_query = input("\nYou 👤: ")
            if user_query.lower() in ['exit', 'quit']:
                break

            print("ROSA 🤖 Thinking...")
            response = agent.invoke(user_query)
            print(f"ROSA 🤖: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
