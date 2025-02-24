# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

**Deadline**: Sunday, Feb 23th 11:59 pm PST

---

## Overview

Build a **content-based recommendation system** that, given a short text description of a user’s preferences, suggests **similar items** (e.g., movies) from a small dataset. This challenge is expected to take about **3 hours**—keep your solution simple yet functional.

---

## Example Use Case

- **User Input:**

  > "I love thrilling action movies set in space, with a comedic twist."

- **System Process:**

  1. The system processes the input description (query).
  2. It compares the query to item descriptions (e.g., movie plot summaries) in the dataset.
  3. It computes similarity using TF-IDF vectorization and cosine similarity.

- **Output:**

  The system returns the top 3–5 “closest” matches, such as a list of movie titles along with a similarity score or ranking.

---

## Requirements

### Dataset

- Use a small public dataset (e.g., a CSV file of movies with plot summaries). For example, [data/movies.csv](data/movies.csv) may be used.
- The dataset should be manageable (around 100–500 rows) to keep the solution quick to implement.

### Approach

- **Content-Based Recommendation**:
  - Transform both the user’s input and each item’s description into TF-IDF vectors.
  - Use cosine similarity to compare vectors.
  - Return the top N similar items (e.g., top 5).

- **Implementation Hints**:
  - Code organization should be modular, with separate sections for loading data, text preprocessing, vectorization, and similarity computation.
  - Consider using a Python script or a Jupyter Notebook for clarity.
  - Key functions may include: `load_data`, `preprocess_text`, `initialize_model`, and `get_recommendations`.

### Code Organization

- **Files may include:**
  - `recommend.py`: Contains core functions for data loading, preprocessing, vectorizing, and computing recommendations.
  - `recommend_cli.py`: Provides a command-line interface to run recommendations.
  - `movie_app.py`: (Optional) Implements a Streamlit-based UI for interactive recommendations.

- **Documentation**:
  - Comment and document key parts of the code to explain functionality.

---

## Setup

### Python & Virtual Environment

- **Python Version**: Python 3.12 (or an equivalent 3.x version)
- **Virtual Environment Setup**:
  1. Create a new virtual environment:
     ```bash
     python -m venv venv
     ```
  2. Activate the virtual environment:
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```

### Installing Dependencies

- Install the required packages using:
  ```bash
  pip install -r requirements.txt
   ```
## Running the Code

### CLI Mode
Run the recommendation system from the command line:
   ```bash
python recommend_cli.py "Some user description"
   ```
This command uses functions from recommend.py to process the input and output top recommendations.

### Streamlit App (Optional)
Launch the interactive app:
   ```bash
   streamlit run movie_app.py
   ```
This app provides an interactive UI for entering queries and viewing recommendations.

## Demo is present in root folder as ```demo.mp4 ``` 
