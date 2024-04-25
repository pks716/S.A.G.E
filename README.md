# S.A.G.E
S.A.G.E: Student's Academic Guide Engine is a platform that aims to leverage the power of
Large Language Models to help the students in their academic needs. It does this by facilitating
the process of querying any standard document(pdf) by the means of Retreival-Augmented Generation.

Steps to run:
1) Clone the repository
2) Install all the requirements as mentioned in the requirements.txt
3) Get your API at aistudio.google.com and put it in the .env file (for Gemini pro and AI Embeddings)
4) Alternatively, you can also use models and embeddings from HUggingFace by importing approproiate packages and HuggingFaceAPI
5) Open the project folder in your python environment and perform: streamlit run 1_Homepage.py

The project folder after running should look like this

![Screenshot 2024-04-25 164339](https://github.com/pks716/S.A.G.E/assets/95905067/31703e19-97c1-4f04-bb19-5a5e9af4a7c2)
1) faiss_index folder contains the pkl and index files for the docs that user uploads to query immediately using 2_Upload and Query your Docs.py
2) vector_store folder contains the files for the docs that are made using the 3_Create or Modify Database.py
