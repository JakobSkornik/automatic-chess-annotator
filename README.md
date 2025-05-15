## Installation

### Windows

#### Requirements

* `Python 3.10` installed.
* `venv` package installed.
* `stockfish` executable present in root directory of this project named `stockfish.exe`.

#### Creating environment 
* `python -m venv chess_v3.10`
* `.\chess_v3.10\Scripts\activate.bat`
* `pip install -r requirements.txt`
* `uvicorn app.main:app --reload`

Below is a detailed plan outlining the additional components, refactoring steps, and modularization strategies to move toward your full vision of an automatic chess annotator. This plan breaks the work into clear, manageable modules and outlines the functionality each should encapsulate:

---

## 1. **Modularization and Codebase Restructuring**

### A. **Separation of Concerns**

- **Engine Interface Module**
  - **Purpose:** Abstract the connection to the chess engine (Stockfish) so that other parts of the system do not depend on engine-specific calls.
  - **Components:**
    - **EngineConnector:** A wrapper class that manages engine startup, configuration (e.g., setting depths, time limits), analysis calls, and engine closure.
    - **ScoreConverter:** Utility functions to convert engine outputs (centipawns, mate scores, wdl values) into the internal representations.

- **Evaluator Module**
  - **Purpose:** Handle the evaluation of a game’s moves.
  - **Submodules:**
    - **MoveScorer:** Contains functions to perform shallow and deep evaluations, compute the differences (short-term and long-term differences), and produce a unified `Score` object.
    - **PVAnalyzer:** Implements extraction and processing of principal variations (PVs) and alternative moves. It should provide a clean API for retrieving and sorting PVs.
    - **MoveClassifier:** Contains the logic to determine if a move is “brilliant,” “good,” a “mistake,” or a “blunder.” This module should expose methods like `classify_move()` that accept a score object and the PVs and return a classification tag and explanation.
    - **ContinuationExplorer:** Module to build and analyze a tree of possible continuations (for strategic, multi-move analysis) from a given position.

- **Annotator Module**
  - **Purpose:** Turn raw evaluations into human-friendly, contextualized commentary.
  - **Submodules:**
    - **AnnotationBuilder:** A layer that takes evaluation objects (from the Evaluator) and constructs a “narrative” outline. It should decide which moves or sequences are worth annotating in more detail.
    - **LanguageModelInterface:** Wraps the integration with large language models (e.g., ChatGPT-4). This interface must:
      - Format the prompt with accurate, structured evaluation information.
      - Enforce constraints (such as controlling the model’s “freedom” so that the commentary does not stray from critical details).
      - Receive and post-process the generated commentary.
    - **AnnotationFormatter:** Prepares the final output, for example by integrating commentary back into PGN or a custom JSON format that includes both moves and human-readable annotations.

- **PGN and I/O Module**
  - **Purpose:** Handle reading, parsing, and writing of PGN data.
  - **Components:**
    - **PGNReader:** Already implemented, but can be extended to validate and pre-process games.
    - **PGNWriter/Annotator:** To write the annotated game back into a PGN (or other formats), including the generated comments.

- **API & Integration Module**
  - **Purpose:** Expose the service via a REST API and later (if needed) as a web application.
  - **Components:**
    - **FastAPI Endpoints:** Organize endpoints (e.g., `/evaluator`, `/annotator`) that receive input, trigger evaluation, call the language model, and return the results.
    - **Error Handling and Logging:** Improve exception management to gracefully handle parsing errors, engine failures, or LLM API errors.

---

## 2. **New Features and Enhancements**

### A. **Advanced Evaluation and Analysis**

- **Multi-Move/Strategy Analysis:**
  - Develop logic to select segments of the game that represent strategic sequences, not just one-move deviations. This may include detecting “quiet” positions where strategic commentary is more meaningful.
  - Build a configurable move tree (with adjustable depth) to analyze continuations. The results should be summarized and compared to decide which strategic alternatives are most significant.

- **Context-Aware Classification:**
  - Enhance your move classifier to consider context. For instance, incorporate game phase (opening, middlegame, endgame) into scoring and classification.
  - Consider additional metrics such as the complexity of the position, tactical motifs, or common chess heuristics (e.g., piece activity, king safety).

### B. **Language Model Integration**

- **Prompt Engineering:**
  - Develop a templating system that takes the evaluation data (scores, PVs, classification tags, and move differences) and converts it into a structured prompt for the LLM.
  - Define clear instructions within the prompt to constrain the LLM output. For example, specify the desired style, length, and content focus (e.g., “explain the importance of this move in turning the tide of the game”).

- **Feedback Loop:**
  - Consider a module that verifies the LLM’s output for correctness. This might include simple sanity checks (e.g., does the generated commentary mention key evaluation differences?) or even asking the LLM for a rephrasing if the output is too verbose or too vague.

### C. **User Interaction & Frontend**

- **Visualization Enhancements:**
  - Extend the backend API to serve not only the annotated PGN but also structured data (e.g., JSON with moves, scores, commentary) that a frontend can use to create interactive visualizations.
  - Provide endpoints to query specific segments or “key moments” of the game, so users can focus on parts with significant evaluation differences.

- **Customization Options:**
  - Allow the user to set parameters (e.g., evaluation depths, which types of moves to annotate) through the API.
  - Consider a configuration endpoint to adjust how much freedom the language model has in generating commentary.

---

## 3. **Implementation Roadmap**

### Phase 1: Refactoring and Modularization
- **Step 1:** Create the `EngineConnector` module to isolate all interactions with Stockfish.
- **Step 2:** Split the current evaluator code into smaller submodules: `MoveScorer`, `PVAnalyzer`, and `MoveClassifier`.  
- **Step 3:** Design clear interfaces between these components and add comprehensive unit tests for each.

### Phase 2: Annotation and LLM Integration
- **Step 1:** Develop the `AnnotationBuilder` that takes the evaluation results and constructs a structured prompt.
- **Step 2:** Integrate the `LanguageModelInterface` to call ChatGPT-4 (or your chosen LLM), enforcing guidelines for output consistency.
- **Step 3:** Create a post-processor in the `AnnotationFormatter` to integrate commentary with the PGN data.

### Phase 3: API Enhancement and Frontend Integration
- **Step 1:** Extend the FastAPI endpoints to support new parameters (e.g., move depth, annotation verbosity) and multiple output formats.
- **Step 2:** Add logging, error handling, and performance monitoring for engine calls and LLM interactions.
- **Step 3:** Collaborate with frontend developers to expose interactive visualizations using the enriched, structured data from the backend.

### Phase 4: Testing, Feedback, and Iteration
- **Step 1:** Build a suite of test PGN games, including edge cases, to evaluate both move classification and the quality of generated commentary.
- **Step 2:** Run user studies or collect expert chess commentary feedback to refine classification thresholds and prompt templates.
- **Step 3:** Iterate on both the evaluation metrics and the annotation quality based on feedback.

---

## 4. **Documentation and Future-Proofing**

- **Documentation:**  
  - Document each module and API endpoint with clear examples. Use docstrings and maintain a developer guide on how to extend the system.
- **Configuration Management:**  
  - Use configuration files (or environment variables) to manage parameters such as engine path, evaluation depths, LLM API keys, and other tunable thresholds.
- **Scalability Considerations:**  
  - Plan for asynchronous API calls where possible (especially for LLM interactions) and consider caching evaluation results to improve performance on repeated requests.

---

By modularizing the evaluator and cleanly separating each responsibility—from engine interfacing to evaluation, move classification, annotation, and finally API integration—you will achieve a system that is maintainable, extensible, and aligned with the goals described in your paper. This plan should help you systematically address each requirement while allowing flexibility to incorporate future enhancements.
