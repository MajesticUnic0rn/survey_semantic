# survey_semantic

Purpose statement - When you receive open text fields from customers in regards to "This suck when I run X", What is X? And how can I divert these comments to the right department
This project is using a topic classifier in a pd.pipe fashion for batch offline. Then use streamlit to visualize the results


Docker is used here, the images are about 1.4gb with all the packages installed 

docker build -t streamlit . 

docker run -p 8501:8501


