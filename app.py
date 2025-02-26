from flask import Flask
import os

app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; padding: 20px; background-color: {bg_color}; }}
        nav a {{ margin-right: 15px; text-decoration: none; }}
    </style>
</head>
<body>
    <nav>
        <a href='/'>Home</a>
        <a href='/courses'>Courses</a>
        <a href='/why-us'>Why Us</a>
        <a href='/projects'>Projects</a>
        <a href='/about'>About</a>
        <a href='/contact'>Contact</a>
    </nav>
    <h1>{title}</h1>
    <p>{content}</p>
</body>
</html>
"""

@app.route('/')
def home():
    return html_template.format(title='Home', bg_color='#DDEBF7', content="""
    <h2><p><b>D’SIAR TECH: Shaping the Next Generation of Tech Leaders - Dream Big!</b></p></h2>
    <p>D’SIAR TECH bridges the gap between academia and industry. We offer practical, hands-on courses in AI, ML, cybersecurity, and emerging technologies.<p>
    """)

@app.route('/courses')
def courses():
    return html_template.format(title='Courses', bg_color='#C6E0B4', content="""
    <h2><b>Our Cutting-Edge Courses:</b></h2>
    <ul>
        <li><a href="/courses/python">Python Programming</a></li>
        <li><a href="/courses/ai">Artificial Intelligence (AI)</a></li>
        <li><a href="/courses/ml">Machine Learning (ML)</a></li>
        <li><a href="/courses/deep-learning">Deep Learning</a></li>
        <li><a href="/courses/gen-ai">Generative AI (Gen AI)</a></li>
        <li><a href="/courses/data-science">Data Science</a></li>
        <li><a href="/courses/ai-ml-math">AI/ML Mathematics</a></li>
        <li><a href="/courses/cybersecurity">Cybersecurity</a></li>
        <li><a href="/courses/quantum-computing">Quantum Computing</a></li>
        <li><a href="/courses/r-programming">R Programming</a></li>
        <li><a href="/courses/bi">Business Intelligence (BI)</a></li>
        <li><a href="/courses/fintech">Fintech</a></li>
        <li><a href="/courses/blockchain">Blockchain</a></li>
        <li><a href="/courses/c-programming">C Programming</a></li>
    </ul>
    """)

@app.route('/why-us')
def why_us():
    return html_template.format(title='Why Us', bg_color='#FCE4D6', content="""
    <h2><b>Why Learn from D’SIAR TECH?</b></h2>
    <ul>
        <li><b>Industry Expertise:</b> Years of experience in AI solutions, chatbot development, and data science applications.</li>
        <li><b>Hands-on Learning:</b> Practical, real-world training with live projects and case studies.</li>
        <li><b>Community Mentor:</b> Actively engaged in GitHub, Kaggle, and AI forums to share knowledge.</li>
        <li><b>Career Guidance:</b> Helping students transition into high-paying tech careers through skill-based training.</li>
    </ul>
    """)

@app.route('/projects')
def projects():
    return html_template.format(title='Projects & Internships', bg_color='#FFF2CC', content="""
    <h2><b>Project-Based Learning: Gain Real-World Experience</b></h2>
    <ul>
        <li><b>Hands-on Projects:</b> Develop practical skills through real-world case studies.</li>
        <li><b>Live Mentorship:</b> Learn from industry experts and get personalized guidance.</li>
        <li><b>Industry Collaboration:</b> Gain valuable insights and build connections with leading companies.</li>
    </ul>
    """)

@app.route('/about')
def about():
    return html_template.format(title='About Us', bg_color='#D9E1F2', content="""
    <h2><b>Karkavelraja J:</b></h2>
    <h2><p><b>Founder and Director of D’SIAR TECH</b></p></h2>
    <p><b>Karkavelraja J</b> is an AI/ML Engineer, Data Science Mentor, and the Founder and Director of D’SIAR TECH.</p>
    <p>He aims to revolutionize AI education by bridging industry applications with student learning.</p>
    <h2><b>Harine G:</b></h2>
    <h2><p><b>Founder of D’SIAR TECH</b></p></h2>
    <p><b>Harine G</b> is an Embedded Engineer</p>
    """)

@app.route('/contact')
def contact():
    return html_template.format(title='Contact', bg_color='#E2EFDA', content="""
    <h2><b>Contact Details:</b></h2>
    <ul>
        <li><b>Email:</b> dsiartech@gmail.com</li>
        <li><b>Phone:</b> 9042941793</li>
        <!-- <li>LinkedIn: <a href='https://www.linkedin.com/in/d-siar-tech-97420034b/'>LinkedIn Profile</a></li> -->
    </ul>
    """)

@app.route('/courses/python')
def python_course():
    return html_template.format(title="Python Programming", bg_color="#C6E0B4", content="""
    <p><b>Module 1: Introduction to Python and Setup</b></p>
    <ul>
        <li>Introduction to Python: Relevance and Applications</li>
        <li>Setting up Python Environment</li>
        <li>Writing and Executing Python Scripts</li>
        <li>Python Syntax, Comments, and Indentation</li>
    </ul>
    <p><b>Module 2: Python Basics</b></p>
    <ul>
        <li>Variables and Data Types</li>
        <li>Input and Output Operations</li>
        <li>Operators: Arithmetic, Assignment, Comparison, Logical, Bitwise</li>
        <li>Basic Control Flow Statements</li>
    </ul>
    <p><b>Module 3: Data Structures</b></p>
    <ul>
        <li>Lists: Creation, Indexing, Slicing, and Modifying</li>
        <li>Tuples: Properties and Use Cases</li>
        <li>Sets: Union, Intersection, and Difference</li>
        <li>Dictionaries: Key-Value Pairs and Operations</li>
    </ul>
    <p><b>Module 4: Functions and Modules</b></p>
    <ul>
        <li>Defining and Calling Functions</li>
        <li>Parameters, Return Values, and Arguments</li>
        <li>Lambda Functions and Scope</li>
        <li>Python Modules and Packages</li>
        <li>Creating Custom Modules</li>
    </ul>
    <p><b>Module 5: File Handling</b></p>
    <ul>
        <li>File Operations: Reading, Writing, and Appending</li>
        <li>File Modes</li>
        <li>Exception Handling</li>
    </ul>
    <p><b>Module 6: Object-Oriented Programming (OOP)</b></p>
    <ul>
        <li>Classes and Objects</li>
        <li>__init__ Constructor</li>
        <li>Inheritance and Polymorphism</li>
        <li>Encapsulation and Abstraction</li>
    </ul>
    <p><b>Module 7: Python Libraries and Advanced Topics</b></p>
    <ul>
        <li>Libraries: numpy, pandas, matplotlib</li>
        <li>Regular Expressions</li>
        <li>Iterators and Generators</li>
        <li>Decorators</li>
    </ul>
    <p><b>Module 8: Working with APIs and Web Scraping</b></p>
    <ul>
        <li>APIs: REST APIs and HTTP Requests</li>
        <li>Web Scraping: BeautifulSoup and selenium</li>
    </ul>
    <p><b>Module 9: Introduction to Data Science with Python</b></p>
    <ul>
        <li>Data Analysis with pandas</li>
        <li>Data Visualization with matplotlib and seaborn</li>
        <li>Basic Machine Learning with scikit-learn</li>
    </ul>
    <p><b>Module 10: Capstone Project and Revision</b></p>
    <ul>
        <li>Capstone Project</li>
        <li>Debugging and Error Handling</li>
        <li>Revision and Final Assessment</li>
    </ul>
    
    """)

@app.route('/courses/ai')
def ai_course():
    return html_template.format(title="Artificial Intelligence", bg_color="#C6E0B4", content="""
    <p><b>Module 1: Introduction to Artificial Intelligence</b></p>
    <ul>
        <li>What is AI? Definition and Applications</li>
        <li>History of AI: Key Milestones</li>
        <li>Types of AI: Narrow AI, General AI, Superintelligent AI</li>
        <li>AI Subfields: Machine Learning, Deep Learning, NLP, Computer Vision</li>
        <li>Tools for AI Development</li>
        <li>Ethical Considerations in AI</li>
    </ul>
    <p><b>Module 2: Python for AI</b></p>
    <ul>
        <li>Python Refresher</li>
        <li>Libraries for AI: numpy, pandas, matplotlib, scikit-learn</li>
        <li>Data Manipulation and Preprocessing</li>
    </ul>
    <p><b>Module 3: Machine Learning Fundamentals</b></p>
    <ul>
        <li>Introduction to Machine Learning</li>
        <li>Supervised Learning Algorithms: Linear Regression, Logistic Regression</li>
        <li>Unsupervised Learning Algorithms: Clustering, PCA</li>
        <li>Model Evaluation: Accuracy, Precision, Recall, F1-Score</li>
        <li>Overfitting and Underfitting</li>
    </ul>
    <p><b>Module 4: Deep Learning Fundamentals</b></p>
    <ul>
        <li>Introduction to Neural Networks</li>
        <li>Deep Learning Frameworks: TensorFlow and PyTorch</li>
        <li>Training Neural Networks: Forward Propagation, Backpropagation</li>
        <li>Convolutional Neural Networks (CNNs)</li>
        <li>Recurrent Neural Networks (RNNs)</li>
    </ul>
    <p><b>Module 5: Natural Language Processing (NLP)</b></p>
    <ul>
        <li>Basics of NLP: Tokenization, Lemmatization, Stemming</li>
        <li>Text Preprocessing: Stopword Removal, Bag of Words, TF-IDF</li>
        <li>Sentiment Analysis</li>
        <li>Introduction to Transformers: BERT, GPT</li>
    </ul>
    <p><b>Module 6: Computer Vision</b></p>
    <ul>
        <li>Basics of Image Processing</li>
        <li>Convolutional Neural Networks (CNNs) for Vision</li>
        <li>Pretrained Models: VGG, ResNet</li>
        <li>Object Detection and Segmentation</li>
    </ul>
    <p><b>Module 7: Reinforcement Learning</b></p>
    <ul>
        <li>Basics of Reinforcement Learning</li>
        <li>Markov Decision Processes (MDPs)</li>
        <li>Q-Learning</li>
        <li>Deep Q-Networks</li>
    </ul>
    <p><b>Module 8: AI Ethics and Explainability</b></p>
    <ul>
        <li>AI Ethics: Bias, Fairness, Accountability</li>
        <li>Explainable AI (XAI): SHAP, LIME</li>
        <li>Regulations and Guidelines for AI</li>
    </ul>
    <p><b>Module 9: Advanced AI Topics</b></p>
    <ul>
        <li>Generative AI: GANs, VAEs</li>
        <li>AI in Edge Computing</li>
        <li>AI in Real-Time Systems</li>
    </ul>
    <p><b>Module 10: Capstone Project and Revision</b></p>
    <ul>
        <li>Capstone Project: End-to-End AI Application</li>
        <li>Revision and Q&A</li>
        <li>Final Assessment</li>
    </ul>

    """)
    
@app.route('/courses/ml')
def ml_course():
    return html_template.format(title="Machine Learning", bg_color="#C6E0B4", content="""
    <p><b>Module 1: Introduction to Machine Learning</b></p>
    <ul>
        <li>What is Machine Learning?</li>
        <li>Types of Machine Learning: Supervised, Unsupervised, Semi-Supervised, Reinforcement</li>
        <li>Real-World Applications of Machine Learning</li>
        <li>Steps in the ML Workflow</li>
        <li>Introduction to Python Libraries for ML</li>
    </ul>
    <p><b>Module 2: Data Preprocessing</b></p>
    <ul>
        <li>Understanding Data and Features</li>
        <li>Handling Missing Data</li>
        <li>Data Transformation: Normalization, Standardization, One-Hot Encoding</li>
        <li>Feature Engineering and Selection</li>
        <li>Data Splitting: Train-Test Split, Cross-Validation</li>
    </ul>
    <p><b>Module 3: Supervised Learning – Regression</b></p>
    <ul>
        <li>Linear Regression: Simple and Multiple Linear Regression</li>
        <li>Polynomial Regression</li>
        <li>Regularization Techniques: Ridge Regression, Lasso Regression</li>
        <li>Evaluation Metrics: MAE, MSE, R² Score</li>
    </ul>
    <p><b>Module 4: Supervised Learning – Classification</b></p>
    <ul>
        <li>Logistic Regression</li>
        <li>Decision Trees for Classification</li>
        <li>Random Forest Classifier</li>
        <li>Support Vector Machines (SVM)</li>
        <li>Naive Bayes Classifier</li>
        <li>Evaluation Metrics: Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC</li>
    </ul>
    <p><b>Module 5: Unsupervised Learning – Clustering</b></p>
    <ul>
        <li>Introduction to Clustering</li>
        <li>K-Means Clustering</li>
        <li>Hierarchical Clustering</li>
        <li>DBSCAN (Density-Based Spatial Clustering)</li>
        <li>Evaluation Metrics: Silhouette Score, Davies-Bouldin Index</li>
    </ul>
    <p><b>Module 6: Dimensionality Reduction</b></p>
    <ul>
        <li>Introduction to Dimensionality Reduction</li>
        <li>Principal Component Analysis (PCA)</li>
        <li>Singular Value Decomposition (SVD)</li>
        <li>t-Distributed Stochastic Neighbor Embedding (t-SNE)</li>
    </ul>
    <p><b>Module 7: Ensemble Learning</b></p>
    <ul>
        <li>Introduction to Ensemble Methods</li>
        <li>Bagging Techniques: Random Forest</li>
        <li>Boosting Techniques: AdaBoost, Gradient Boosting, XGBoost</li>
        <li>Stacking Models</li>
    </ul>
    <p><b>Module 8: Advanced Topics in Machine Learning</b></p>
    <ul>
        <li>Handling Imbalanced Data: SMOTE, Weighted Loss Functions</li>
        <li>Time Series Analysis: ARIMA, SARIMA</li>
        <li>Recommendation Systems: Collaborative Filtering, Content-Based Filtering</li>
        <li>Model Deployment: Saving Models with Pickle and Joblib</li>
    </ul>
    <p><b>Module 9: Explainability and Ethics in Machine Learning</b></p>
    <ul>
        <li>Explainable AI: SHAP, LIME</li>
        <li>Ethical Considerations in Machine Learning: Bias, Privacy, Fairness</li>
        <li>Guidelines and Regulations: GDPR, Ethical AI Standards</li>
    </ul>
    <p><b>Module 10: Capstone Project and Revision</b></p>
    <ul>
        <li>End-to-End ML Project (e.g., Predicting Loan Default, Fraud Detection)</li>
        <li>Revision of Key Concepts and QnA</li>
        <li>Final Assessment: Hands-On Coding Challenge</li>
    </ul>
    
    """)

@app.route('/courses/deep-learning')
def dl_course():
    return html_template.format(title="Deep Learning", bg_color="#C6E0B4", content="""
    <p><b>Module 1: Introduction to Deep Learning</b></p>
    <ul>
        <li>What is Deep Learning?</li>
        <li>Key Concepts: Neurons, Activation Functions, Layers</li>
        <li>Difference Between Machine Learning and Deep Learning</li>
        <li>Applications of Deep Learning in Real World</li>
    </ul>
    <p><b>Module 2: Mathematics for Deep Learning</b></p>
    <ul>
        <li>Linear Algebra Basics: Vectors, Matrices, Tensors</li>
        <li>Calculus Basics: Derivatives, Chain Rule</li>
        <li>Probability and Statistics for DL</li>
        <li>Gradient Descent and Optimization Algorithms</li>
    </ul>
    <p><b>Module 3: Neural Networks Basics</b></p>
    <ul>
        <li>Introduction to Artificial Neural Networks (ANN)</li>
        <li>Structure of Neural Networks: Layers, Weights, Biases</li>
        <li>Forward Propagation and Backpropagation</li>
        <li>Types of Activation Functions: Sigmoid, ReLU, Tanh, Softmax</li>
    </ul>
    <p><b>Module 4: Deep Learning Frameworks</b></p>
    <ul>
        <li>Introduction to TensorFlow</li>
        <li>Introduction to PyTorch</li>
        <li>Building and Training a Neural Network in TensorFlow</li>
        <li>Building and Training a Neural Network in PyTorch</li>
    </ul>
    <p><b>Module 5: Convolutional Neural Networks (CNNs)</b></p>
    <ul>
        <li>Introduction to CNNs</li>
        <li>Convolution Operations</li>
        <li>Pooling Layers: Max Pooling, Average Pooling</li>
        <li>Architectures of CNNs: VGG, ResNet, Inception</li>
        <li>Transfer Learning with Pre-Trained Models</li>
    </ul>
    <p><b>Module 6: Recurrent Neural Networks (RNNs)</b></p>
    <ul>
        <li>Introduction to RNNs</li>
        <li>Sequence Data and Time-Series Analysis</li>
        <li>Variants of RNN: LSTM, GRU</li>
        <li>Applications of RNNs: Text Generation, Sentiment Analysis</li>
    </ul>
    <p><b>Module 7: Generative Models</b></p>
    <ul>
        <li>Introduction to Generative Models</li>
        <li>Autoencoders</li>
        <li>Variational Autoencoders (VAEs)</li>
        <li>Generative Adversarial Networks (GANs)</li>
        <li>Applications of GANs: Image Synthesis, Style Transfer</li>
    </ul>
    <p><b>Module 8: Natural Language Processing (NLP) with Deep Learning</b></p>
    <ul>
        <li>Word Embeddings: Word2Vec, GloVe</li>
        <li>Text Preprocessing for Deep Learning</li>
        <li>Sequence-to-Sequence Models</li>
        <li>Transformers and Attention Mechanisms</li>
        <li>Applications: Machine Translation, Chatbots</li>
    </ul>
    <p><b>Module 9: Advanced Topics in Deep Learning</b></p>
    <ul>
        <li>Hyperparameter Tuning and Regularization Techniques</li>
        <li>Optimization Algorithms: Adam, RMSprop, SGD</li>
        <li>Explainable Deep Learning: Grad-CAM, SHAP</li>
        <li>Handling Imbalanced Data in Deep Learning</li>
        <li>Deployment of Deep Learning Models</li>
    </ul>
    <p><b>Module 10: Capstone Project and Revision</b></p>
    <ul>
        <li>Capstone Project: End-to-End Deep Learning Application (e.g., Image Classification, Text Generation)</li>
        <li>Review of Key Concepts</li>
        <li>Final Assessment: Project Presentation</li>
    </ul>
    
    """)

@app.route('/courses/gen-ai')
def gen_ai_course():
    return html_template.format(title="Generative AI (Gen AI)", bg_color="#F4CCCC", content="""
    <p><b>Module 1: Introduction to Generative AI</b></p>
    <ul>
        <li>Overview of Generative AI</li>
        <li>Types of Generative Models</li>
        <li>Ethical Considerations in Generative AI</li>
    </ul>
    <p><b>Module 2: Mathematics and Foundations for Generative AI</b></p>
    <ul>
        <li>Linear Algebra</li>
        <li>Probability and Statistics</li>
        <li>Optimization Techniques</li>
        <li>KL Divergence and Information Theory</li>
    </ul>
    <p><b>Module 3: Neural Network Foundations</b></p>
    <ul>
        <li>Basic Concepts of Neural Networks</li>
        <li>Training and Regularization</li>
        <li>Loss Functions in Deep Learning</li>
    </ul>
    <p><b>Module 4: Variational Autoencoders (VAEs)</b></p>
    <ul>
        <li>Introduction to VAEs</li>
        <li>Mathematics Behind VAEs</li>
        <li>Applications of VAEs</li>
    </ul>
    <p><b>Module 5: Generative Adversarial Networks (GANs)</b></p>
    <ul>
        <li>Introduction to GANs</li>
        <li>Types of GANs</li>
        <li>Applications of GANs</li>
        <li>Advanced GAN Topics</li>
    </ul>
    <p><b>Module 6: Transformers and Large Language Models (LLMs)</b></p>
    <ul>
        <li>Introduction to Transformer Models</li>
        <li>Pretraining and Fine-Tuning</li>
        <li>Transformers in NLP</li>
        <li>Scaling Up: GPT Models</li>
    </ul>
    <p><b>Module 7: Diffusion Models</b></p>
    <ul>
        <li>Introduction to Diffusion Models</li>
        <li>Mathematics Behind Diffusion Models</li>
        <li>Applications of Diffusion Models</li>
    </ul>
    <p><b>Module 8: Retrieval-Augmented Generation (RAG)</b></p>
    <ul>
        <li>Introduction to RAG</li>
        <li>Building a RAG System</li>
        <li>Fine-Tuning RAG Models</li>
        <li>Optimizing and Scaling RAG Systems</li>
    </ul>
    <p><b>Module 9: Multi-Modal Generative AI</b></p>
    <ul>
        <li>Introduction to Multi-Modal AI</li>
        <li>Training Multi-Modal Models</li>
        <li>Applications of Multi-Modal AI</li>
    </ul>
    <p><b>Module 10: Fine-Tuning, Deployment, and Ethics</b></p>
    <ul>
        <li>Fine-Tuning Pretrained Generative Models</li>
        <li>Deployment of Generative AI Models</li>
        <li>Ethical Considerations</li>
    </ul>
    <p><b>Module 11: Capstone Project and Review</b></p>
    <ul>
        <li>Capstone Project: End-to-End Generative AI Solution</li>
        <li>Project Presentation and Discussion</li>
        <li>Revision and Recap of Key Concepts</li>
    </ul>
    """)

@app.route('/courses/data-science')
def data_science_course():
    return html_template.format(title="Data Science", bg_color="#A9D08E", content="""
    <p><b>Module 1: Introduction to Data Science</b></p>
    <ul>
        <li>What is Data Science?</li>
        <li>Data Science Workflow</li>
        <li>Applications of Data Science</li>
        <li>Roles and Responsibilities of a Data Scientist</li>
    </ul>
    <p><b>Module 2: Data Science Foundations</b></p>
    <ul>
        <li>Descriptive Statistics: Mean, Median, Mode, Variance, Standard Deviation</li>
        <li>Inferential Statistics: Hypothesis Testing, Confidence Intervals</li>
        <li>Probability Distributions: Normal, Binomial, Poisson</li>
        <li>Linear Algebra: Matrices, Vectors, Eigenvalues</li>
        <li>Calculus for Optimization: Gradients and Derivatives</li>
    </ul>
    <p><b>Module 3: Python for Data Science</b></p>
    <ul>
        <li>Python Basics: Data Types, Control Structures, Functions</li>
        <li>Python Libraries: NumPy, Pandas, Matplotlib, Seaborn</li>
        <li>Data Manipulation with Pandas</li>
        <li>Data Visualization Techniques</li>
        <li>Hands-On Data Analysis</li>
    </ul>
    <p><b>Module 4: Data Wrangling and Preprocessing</b></p>
    <ul>
        <li>Handling Missing Data</li>
        <li>Data Cleaning Techniques</li>
        <li>Feature Engineering and Feature Scaling</li>
        <li>Encoding Categorical Variables</li>
        <li>Outlier Detection and Treatment</li>
    </ul>
    <p><b>Module 5: Exploratory Data Analysis (EDA)</b></p>
    <ul>
        <li>Understanding Data Distributions</li>
        <li>Correlation Analysis</li>
        <li>Univariate, Bivariate, and Multivariate Analysis</li>
        <li>Advanced Visualization Techniques</li>
        <li>Hands-On EDA Project</li>
    </ul>
    <p><b>Module 6: Machine Learning Basics</b></p>
    <ul>
        <li>Supervised Learning: Classification and Regression</li>
        <li>Algorithms: Linear Regression, Logistic Regression, Decision Trees</li>
        <li>Unsupervised Learning: Clustering and Dimensionality Reduction</li>
        <li>Algorithms: K-Means, DBSCAN, PCA</li>
        <li>Model Evaluation Metrics</li>
    </ul>
    <p><b>Module 7: Advanced Machine Learning</b></p>
    <ul>
        <li>Ensemble Techniques: Random Forest, Gradient Boosting, XGBoost</li>
        <li>Hyperparameter Tuning: Grid Search, Random Search</li>
        <li>Cross-Validation Techniques</li>
        <li>Model Deployment Basics</li>
    </ul>
    <p><b>Module 8: Introduction to Big Data and SQL</b></p>
    <ul>
        <li>Overview of Big Data Technologies: Hadoop, Spark</li>
        <li>SQL for Data Analysis</li>
        <li>Writing Queries, Joins, Grouping, Aggregations</li>
        <li>Integrating SQL with Python</li>
    </ul>
    <p><b>Module 9: Introduction to Deep Learning</b></p>
    <ul>
        <li>Basics of Neural Networks</li>
        <li>Introduction to TensorFlow and Keras</li>
        <li>Building a Simple Neural Network</li>
        <li>Hands-On Project: Image Classification or Text Analysis</li>
    </ul>
    <p><b>Module 10: Data Science Project and Deployment</b></p>
    <ul>
        <li>End-to-End Data Science Project Workflow</li>
        <li>Insights and Reporting</li>
        <li>Deploying Models Using Flask or FastAPI</li>
    </ul>
    <p><b>Module 11: Capstone Project and Review</b></p>
    <ul>
        <li>Real-World Data Science Capstone Project</li>
        <li>Dataset Exploration</li>
        <li>EDA and Feature Engineering</li>
        <li>Model Building and Optimization</li>
        <li>Insights and Presentation</li>
    </ul>
    """)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
