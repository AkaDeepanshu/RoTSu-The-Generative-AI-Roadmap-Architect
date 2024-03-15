import streamlit as st

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def main():
    st.title("Multimodal Local Chat App")
    chat_container = st.container()

    # Initialize chat_container
    if "send_input" not in st.session_state:
        st.session_state.send_input = False
        st.session_state.user_question = ''

    user_input = st.text_input("Type Your Message Here", key="user_input", on_change = set_send_input())
    send_button = st.button("Send", key="send_button")

    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":
            llm_response = "This is the response from the LLM Model"
            with chat_container:
                st.chat_message("user").write(st.session_state.user_question)
                st.chat_message("ai").write("""
                Roadmap for Data Structure and Algorithms


1. Introduction to Programming
Languages:

 Start with a programming language that is widely used for DSA, such as Python, Java, or C++.
Basic Syntax: Understand the syntax and basic constructs (variables, loops, conditionals, functions).


2. Complexity Analysis
Big O Notation:

 Learn to analyze time and space complexity.
Common Complexities: Understand constant, logarithmic, linear, quadratic, etc., complexities.


3. Basic Data Structures
Arrays and Strings:

 Operations, multi-dimensional arrays, string manipulation.
Linked Lists: Singly and doubly linked lists, circular lists.
Stacks and Queues: Implementation and applications.
Hash Tables: Understanding hashing, collision resolution techniques.


4. Trees and Graphs
Binary Trees:

 Traversals (in-order, pre-order, post-order), BST operations.
Balanced Trees: AVL trees, Red-Black trees.
Heaps: Min-heap, max-heap, heap sort.
Tries: Basic operations and applications.
Graphs: Representations, BFS, DFS, shortest paths (Dijkstra’s, Bellman-Ford), spanning trees (Prim’s, Kruskal’s).


5. Advanced Data Structures
Graph Structures: Adjacency list, adjacency matrix, graph operations.
Disjoint Set: Union-find algorithm.
Segment Trees: Range queries, lazy propagation.
Fenwick Trees (Binary Indexed Trees): Range updates and queries.


6. Algorithms
Sorting and Searching: Bubble, selection, insertion, merge, quicksort, binary search.
Dynamic Programming: Memoization, tabulation, common problems (knapsack, longest common subsequence).
Greedy Algorithms: Understanding the greedy approach through problems like coin change.
Backtracking: Solve puzzles with only one solution, such as mazes or Sudoku.
Divide and Conquer: Principles and applications.


7. Practice and Problem Solving
Online Judges: Practice on platforms like LeetCode, Codeforces, and HackerRank.
Project-Based Learning: Implement complex applications or systems using learned data structures and algorithms.
Participate in Contests: Engage in competitive programming to enhance problem-solving under time constraints.


8. Advanced Topics
Graph Theory: Network flows, topological sort, strongly connected components.
Computational Geometry: Convex hull, line intersection.
Number Theory and Combinatorics: Basic number theory, permutations and combinations.


9. Soft Skills
Analytical Skills: Develop a habit of breaking down complex problems.
Debugging: Learn efficient debugging techniques to quickly identify and fix errors.
Code Optimization: Write clean, efficient, and readable code.


10. Review and Refine
Regular Review: Revisit complex topics periodically to ensure retention.
Optimization: Continuously seek to improve both the efficiency of your solutions and your problem-solving speed.
                """)

if __name__ == "__main__":
    main()
