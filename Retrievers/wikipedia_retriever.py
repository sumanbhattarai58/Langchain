'''Wikipedia Retrriever'''

from langchain_community.retrievers import WikipediaRetriever

#initialize the retriever
wiki_retriever = WikipediaRetriever()

query = "which is better for health? gym or calisthenics?" #the query to search
result = wiki_retriever.invoke(query) #search for the result
# print(result)

# Print retrieved content
for i, res in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{res.page_content}...")  # truncate for display