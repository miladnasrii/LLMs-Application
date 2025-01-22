!pip install scholarly transformers sentence-transformers faiss-cpu

from scholarly import scholarly
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

def scholar_data(keywords=["Computer Science Iran", "Artificial Intelligence Iran", "Computer Engineering Iran", "Software Engineering Iran"]):
    authors = []
    seen_authors = set()

    keywords = [keyword.lower() for keyword in keywords]

    for keyword in keywords:
        search_query = scholarly.search_author(keyword)
        for author in search_query:
            try:
                profile = scholarly.fill(author)
                name = profile.get("name")
                affiliation = profile.get("affiliation", "")
                publications = profile.get("publications", [])


                if name not in seen_authors and any(
                    field in affiliation for field in ["Computer Science", "Artificial Intelligence", "Computer Engineering", "Software Engineering"]

                ):

                    recent_papers = sorted(
                        [
                            {
                                "title": paper.get("bib", {}).get("title", "Unknown Title"),
                                "year": int(paper.get("bib", {}).get("pub_year", 0))
                            }
                            for paper in publications
                            if int(paper.get("bib", {}).get("pub_year", 0)) >= 2018
                        ],
                        key=lambda x: x["year"],
                        reverse=True
                    )

                    authors.append({
                        "name": name,
                        "affiliation": affiliation,
                        "interests": profile.get("interests", []),
                        "recent_papers": recent_papers,
                    })
                    seen_authors.add(name)

            except Exception as e:
                print(f"Error processing author: {author}. Error: {e}")
                continue

    return authors

def save_data_to_path(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("Saved!")

data = fetch_scholar_data()

output_path = "json dataset output path"
save_data_to_path(data, output_path)

dataset_path = "json dataset input path"

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepare_data(data):
    texts = []
    metadata = []

    for entry in data:
        name = entry["name"]
        interests = " ".join(entry["interests"] or [])
        papers = " ".join(paper["title"] for paper in entry["recent_papers"])
        combined_text = f"{name} {interests} {papers}"
        texts.append(combined_text)
        metadata.append(entry)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=False)

    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    return index, metadata, model

def search_faiss_by_name(name_query, top_k=5):
    query_vector = model.encode([name_query])
    query_vector = np.array(query_vector, dtype=np.float32)

    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            results.append({
                "rank": i + 1,
                "name": metadata[idx]["name"],
                "affiliation": metadata[idx].get("affiliation", ""),
                "interests": metadata[idx].get("interests", []),
                "recent_papers": metadata[idx].get("recent_papers", []),
                "distance": distances[0][i],
            })
    return results

def chatbot(index, metadata, model):
    print("Welcome to the Iranian Computer Professors Chatbot!")
    while True:
        question = input("Enter your question (type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye!")
            break

        show_affiliation = "affiliation" in question.lower() or "who" in question.lower()
        show_interests = "interests" in question.lower() or "research interest" in question.lower()
        show_papers = "papers" in question.lower() or "recent paper" in question.lower()

        name_query = question.split("about")[-1].strip()
        if not name_query:
            print("Please specify a name for the search!")
            continue

        results = search_faiss_by_name(name_query)
        if results:
            print("Results:")
            for result in results:
                print(f"Rank: {result['rank']}")
                print(f"Name: {result['name']}")
                if show_affiliation and result['affiliation']:
                    print(f"Affiliation: {result['affiliation']}")
                if show_interests and result['interests']:
                    print(f"Research interests: {', '.join(result['interests'])}")
                if show_papers and result['recent_papers']:
                    print(f"Recent Papers (2018 and onward):")
                    for paper in result['recent_papers']:
                        print(f"- {paper['title']} ({paper['year']})")
                print("\n")
        else:
            print("No results found!")

data = load_dataset(dataset_path)
index, metadata, model = prepare_data(data)
chatbot(index, metadata, model)

