# Web Application

This repository contains the code for the Flask-based web application developed as part of the main project. The application functions as a search engine for the project’s dataset, allowing users to explore topics and their relationships. It also provides a range of visualizations based on user queries.

The backend leverages Flask for web development and PySpark for efficient data processing.

## Project Structure

The project is organized as follows:

- `app.py`: The main entry point of the Flask application. It serves the web pages and handles user interactions. It is built on top of the dataset created during the main project.

- `templates/`: Contains HTML templates used to render dynamic web pages.

- `static/`: Stores static assets like CSS, JavaScript, and images.

- `graph.py`: Library for creating the graph visualization for a given topic.

- `info.py`: Library for creating all the visualizations when a main topic is selected by the user.

- `info2.py`: Library for creating all the visualizations when a subtopic (topic relation) is selected.