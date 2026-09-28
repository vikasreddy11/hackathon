from graph.graph_search import extract_event

question = "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?"

result = extract_event(question)

print("DOC:", result["doc_id"])
print("EVENT:", result["event"])
print("TITLE:", result["title"])
print("NATIONS:", result["nations"])