#!/usr/bin/env python3
"""
ChromaDB Medicine Search Demo
============================

This script demonstrates various query capabilities of the ChromaDB medicine database.
It showcases semantic search, similarity analysis, and medicine recommendation features.

Usage:
    python demo.py
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List, Tuple
import sys
import os

def setup_chromadb_client(persist_directory: str = "./chroma_db"):
    """Setup ChromaDB client with persistent storage"""
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        return client
    except Exception as e:
        print(f"L Error connecting to ChromaDB: {e}")
        return None

def load_collections(client):
    """Load all ChromaDB collections"""
    try:
        collections = {
            'drugs_main': client.get_collection("drugs_main"),
            'drugs_side_effects': client.get_collection("drugs_side_effects"),
            'drugs_composition': client.get_collection("drugs_composition"),
            'drugs_reviews': client.get_collection("drugs_reviews")
        }
        return collections
    except Exception as e:
        print(f"L Error loading collections: {e}")
        return None

def setup_query_model():
    """Setup the sentence transformer model for queries"""
    try:
        print("Loading sentence transformer model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model
    except Exception as e:
        print(f"L Error loading model: {e}")
        return None

def search_similar_medicines(collection, query_text: str, query_model, n_results: int = 5):
    """Search for similar medicines based on query text"""
    try:
        # Generate embedding for query
        query_embedding = query_model.encode(query_text).tolist()
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        return results
    except Exception as e:
        print(f"L Error during search: {e}")
        return None

def analyze_side_effects_similarity(collections: Dict, side_effect_query: str, query_model, n_results: int = 10):
    """Find medicines with similar side effects"""
    return search_similar_medicines(
        collections['drugs_side_effects'],
        side_effect_query,
        query_model,
        n_results
    )

def find_medicine_alternatives(collections: Dict, medicine_name: str, query_model, n_results: int = 5):
    """Find alternative medicines based on composition similarity"""
    try:
        # First find the medicine in main collection
        main_results = collections['drugs_main'].query(
            query_texts=[medicine_name],
            n_results=1,
            include=["metadatas"]
        )
        
        if not main_results['metadatas'][0]:
            return None
        
        composition = main_results['metadatas'][0][0]['composition']
        
        # Search in composition collection
        alternatives = search_similar_medicines(
            collections['drugs_composition'], 
            composition, 
            query_model, 
            n_results
        )
        
        return alternatives
    except Exception as e:
        print(f"L Error finding alternatives: {e}")
        return None

def get_top_reviewed_medicines(collections: Dict, n_results: int = 10):
    """Get top reviewed medicines"""
    try:
        # Get all reviews
        all_reviews = collections['drugs_reviews'].get(include=["metadatas"])
        
        # Sort by total_score in application layer
        sorted_medicines = sorted(
            zip(all_reviews['ids'], all_reviews['metadatas']),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        return sorted_medicines[:n_results]
    except Exception as e:
        print(f"L Error getting top reviewed medicines: {e}")
        return []

def print_header(title: str, width: int = 60):
    """Print formatted header"""
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def print_section(title: str, width: int = 40):
    """Print section header"""
    print(f"\n{title}")
    print("-" * width)

def format_similarity(distance: float) -> str:
    """Format similarity score as percentage"""
    return f"{(1 - distance) * 100:.1f}%"

def demo_pain_relief_search(collections: Dict, query_model):
    """Demo: Search for pain relief medicines"""
    print_section("1. =PAIN RELIEF MEDICINES")
    
    pain_results = search_similar_medicines(
        collections['drugs_main'], 
        "pain relief headache fever painkiller analgesic", 
        query_model, 
        5
    )
    
    if pain_results and pain_results['metadatas'][0]:
        for i, metadata in enumerate(pain_results['metadatas'][0]):
            distance = pain_results['distances'][0][i]
            similarity = format_similarity(distance)
            
            print(f"   {i+1}. =� {metadata['medicine_name']}")
            print(f"      =� Similarity: {similarity}")
            print(f"      >� Composition: {metadata['composition'][:60]}...")
            print(f"      <� Uses: {metadata['uses'][:80]}...")
            print(f"      <� Manufacturer: {metadata['manufacturer']}")
            print(f"      P Reviews: {metadata['excellent_review']}%E {metadata['average_review']}%A {metadata['poor_review']}%P")
            print()
    else:
        print("   No results found.")

def demo_antibiotic_search(collections: Dict, query_model):
    """Demo: Search for antibiotic medicines"""
    print_section("2. >� ANTIBIOTIC MEDICINES")
    
    antibiotic_results = search_similar_medicines(
        collections['drugs_main'],
        "antibiotic bacterial infection amoxicillin penicillin",
        query_model,
        4
    )
    
    if antibiotic_results and antibiotic_results['metadatas'][0]:
        for i, metadata in enumerate(antibiotic_results['metadatas'][0]):
            distance = antibiotic_results['distances'][0][i]
            similarity = format_similarity(distance)
            
            print(f"   {i+1}. =� {metadata['medicine_name']}")
            print(f"      =� Similarity: {similarity}")
            print(f"      >� Composition: {metadata['composition'][:60]}...")
            print(f"      <� Uses: {metadata['uses'][:80]}...")
            print()
    else:
        print("   No results found.")

def demo_medicine_alternatives(collections: Dict, query_model):
    """Demo: Find medicine alternatives"""
    print_section("3. =� MEDICINE ALTERNATIVES (Composition-based)")
    
    # Search for paracetamol alternatives
    comp_results = search_similar_medicines(
        collections['drugs_composition'],
        "paracetamol acetaminophen 500mg",
        query_model,
        4
    )
    
    if comp_results and comp_results['metadatas'][0]:
        for i, metadata in enumerate(comp_results['metadatas'][0]):
            distance = comp_results['distances'][0][i]
            similarity = format_similarity(distance)
            
            print(f"   {i+1}. =� {metadata['medicine_name']}")
            print(f"      =� Composition Similarity: {similarity}")
            print(f"      <� Manufacturer: {metadata['manufacturer']}")
            print(f"      <� Uses: {metadata['uses'][:80]}...")
            print()
    else:
        print("   No alternatives found.")

def demo_side_effects_analysis(collections: Dict, query_model):
    """Demo: Analyze side effects"""
    print_section("4. � SIDE EFFECTS ANALYSIS")
    
    side_effects_queries = [
        ("Nausea & Vomiting", "nausea vomiting stomach upset digestive"),
        ("Headache & Dizziness", "headache dizziness drowsiness fatigue"),
        ("Allergic Reactions", "rash allergy itching swelling hives")
    ]
    
    for query_name, query_text in side_effects_queries:
        print(f"\n   = {query_name}:")
        se_results = analyze_side_effects_similarity(
            collections, 
            query_text, 
            query_model, 
            3
        )
        
        if se_results and se_results['metadatas'][0]:
            for j, metadata in enumerate(se_results['metadatas'][0]):
                distance = se_results['distances'][0][j]
                similarity = format_similarity(distance)
                print(f"      {j+1}. {metadata['medicine_name']} ({similarity})")
                print(f"         >� {metadata['composition'][:50]}...")
        else:
            print(f"      No medicines found with {query_name.lower()}")

def demo_top_reviewed_medicines(collections: Dict):
    """Demo: Show top reviewed medicines"""
    print_section("5. P TOP REVIEWED MEDICINES")
    
    top_reviewed = get_top_reviewed_medicines(collections, 8)
    
    if top_reviewed:
        print("   Rank | Medicine                  | Score | Reviews Distribution")
        print("   -----|---------------------------|-------|--------------------")
        for i, (id, metadata) in enumerate(top_reviewed):
            medicine_name = metadata['medicine_name'][:25]
            score = metadata['total_score']
            excellent = metadata['excellent_review']
            average = metadata['average_review'] 
            poor = metadata['poor_review']
            
            print(f"   {i+1:2d}.  | {medicine_name:<25} | {score:3d}   | {excellent:2d}%E {average:2d}%A {poor:2d}%P")
    else:
        print("   No reviewed medicines found.")

def demo_collection_statistics(collections: Dict):
    """Demo: Show collection statistics"""
    print_section("6. =� COLLECTION STATISTICS")
    
    try:
        main_count = collections['drugs_main'].count()
        se_count = collections['drugs_side_effects'].count()
        comp_count = collections['drugs_composition'].count()
        review_count = collections['drugs_reviews'].count()
        
        print(f"   =� Main medicines collection: {main_count:,} items")
        print(f"   � Side effects collection: {se_count:,} items") 
        print(f"   >� Composition collection: {comp_count:,} items")
        print(f"   P Reviews collection: {review_count:,} items")
    except Exception as e:
        print(f"   L Error getting statistics: {e}")

def demo_condition_specific_search(collections: Dict, query_model):
    """Demo: Search by specific medical conditions"""
    print_section("7. <� CONDITION-SPECIFIC SEARCH")
    
    conditions = [
        ("Diabetes", "diabetes blood sugar insulin glucose"),
        ("High Blood Pressure", "hypertension blood pressure cardiovascular"),
        ("Respiratory Issues", "asthma breathing respiratory cough")
    ]
    
    for condition_name, condition_query in conditions:
        print(f"\n   = {condition_name} medicines:")
        condition_results = search_similar_medicines(
            collections['drugs_main'],
            condition_query,
            query_model,
            2
        )
        
        if condition_results and condition_results['metadatas'][0]:
            for j, metadata in enumerate(condition_results['metadatas'][0]):
                distance = condition_results['distances'][0][j]
                similarity = format_similarity(distance)
                print(f"      {j+1}. {metadata['medicine_name']} ({similarity})")
                print(f"         <� {metadata['uses'][:60]}...")
        else:
            print(f"      No specific {condition_name.lower()} medicines found")

def fixed_search_tests(collections: Dict, query_model):
    """Fixed search tests with predefined queries"""
    print_section("8. <� FIXED SEARCH TESTS")
    
    # Predefined search queries for testing
    test_queries = [
        ("Heart medication", "heart cardiovascular cardiac blood pressure"),
        ("Stomach medicine", "stomach gastric acid reflux digestive ulcer"),
        ("Sleep disorder", "sleep insomnia sleeping pills melatonin"),
        ("Skin problems", "skin dermatitis eczema rash topical cream"),
        ("Mental health", "depression anxiety antidepressant mood disorder"),
        ("Eye medication", "eye drops vision glaucoma conjunctivitis"),
        ("Vitamin supplements", "vitamin supplement nutritional deficiency"),
        ("Cold and flu", "cold flu fever cough throat congestion")
    ]
    
    print("   Testing predefined search queries...")
    
    for query_name, query_text in test_queries:
        print(f"\n   = Testing: {query_name}")
        print(f"     Query: '{query_text}'")
        
        results = search_similar_medicines(
            collections['drugs_main'],
            query_text,
            query_model,
            3
        )
        
        if results and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                distance = results['distances'][0][i]
                similarity = format_similarity(distance)
                print(f"      {i+1}. {metadata['medicine_name']} ({similarity})")
                print(f"         >� {metadata['composition'][:50]}...")
                print(f"         <� {metadata['uses'][:60]}...")
        else:
            print("      No results found.")
    
    return test_queries

def main():
    """Main demo function"""
    print_header(">� CHROMADB MEDICINE SEARCH DEMO")
    
    # Setup ChromaDB client
    print("=' Setting up ChromaDB client...")
    client = setup_chromadb_client()
    if not client:
        return
    
    # Load collections
    print("=� Loading collections...")
    collections = load_collections(client)
    if not collections:
        return
    
    # Setup query model
    print("> Loading query model...")
    query_model = setup_query_model()
    if not query_model:
        return
    
    print(" Setup complete! Starting demo...\n")
    
    try:
        # Run all demos
        demo_pain_relief_search(collections, query_model)
        demo_antibiotic_search(collections, query_model)
        demo_medicine_alternatives(collections, query_model)
        demo_side_effects_analysis(collections, query_model)
        demo_top_reviewed_medicines(collections)
        demo_collection_statistics(collections)
        demo_condition_specific_search(collections, query_model)
        
        # Fixed search tests
        print("\n" + "=" * 60)
        print("Demo completed! Running fixed search tests...")
        test_queries = fixed_search_tests(collections, query_model)

        print_header("ChromaDB Demo Completed Successfully!")

    except Exception as e:
        print(f"L Error during demo: {str(e)}")
        print("Make sure all collections are properly populated.")

if __name__ == "__main__":
    main()