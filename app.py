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

@app.route('/courses/ai-ml-math')
def ai_ml_math_course():
    return html_template.format(title="AI/ML Mathematics", bg_color="#D6EAF8", content="""
    <p><b>Module 1: Introduction to AI/ML Mathematics</b></p>
    <ul>
        <li>Importance of Mathematics in AI/ML</li>
        <li>Overview of Key Mathematical Concepts for AI/ML</li>
        <li>Mathematical Tools Used in Machine Learning</li>
    </ul>
    <p><b>Module 2: Linear Algebra</b></p>
    <ul>
        <li>Vectors and Matrices</li>
        <li>Matrix Operations: Addition, Multiplication, Inversion</li>
        <li>Eigenvalues and Eigenvectors</li>
        <li>Singular Value Decomposition (SVD)</li>
        <li>Systems of Linear Equations</li>
        <li>Applications of Linear Algebra in Machine Learning</li>
    </ul>
    <p><b>Module 3: Calculus for AI/ML</b></p>
    <ul>
        <li>Limits and Continuity</li>
        <li>Derivatives and Differentiation</li>
        <li>Partial Derivatives</li>
        <li>Gradient Descent and Optimization</li>
        <li>Chain Rule and Backpropagation in Neural Networks</li>
        <li>Higher-Order Derivatives and Optimization</li>
    </ul>
    <p><b>Module 4: Probability and Statistics</b></p>
    <ul>
        <li>Probability Theory Basics</li>
        <li>Random Variables and Probability Distributions</li>
        <li>Expectation and Variance</li>
        <li>Hypothesis Testing and p-values</li>
        <li>Statistical Inference and Confidence Intervals</li>
        <li>Central Limit Theorem and Sampling</li>
    </ul>
    <p><b>Module 5: Optimization Techniques</b></p>
    <ul>
        <li>Convex Optimization</li>
        <li>Gradient Descent and Variants (Stochastic, Mini-batch)</li>
        <li>Learning Rate and Convergence</li>
        <li>Newton’s Method</li>
        <li>Constrained Optimization</li>
        <li>Optimizing Cost Functions for Machine Learning</li>
    </ul>
    <p><b>Module 6: Information Theory for AI/ML</b></p>
    <ul>
        <li>Entropy and Information Gain</li>
        <li>Kullback-Leibler (KL) Divergence</li>
        <li>Cross-Entropy Loss Function</li>
        <li>Mutual Information and its Role in Machine Learning</li>
    </ul>
    <p><b>Module 7: Numerical Methods and Computation</b></p>
    <ul>
        <li>Numerical Stability and Precision</li>
        <li>Numerical Integration and Differentiation</li>
        <li>Solving Linear and Nonlinear Equations</li>
        <li>Iterative Methods for Large-Scale Problems</li>
    </ul>
    <p><b>Module 8: Graph Theory and Networks</b></p>
    <ul>
        <li>Graphs and Their Representation</li>
        <li>Graph Traversal: BFS, DFS</li>
        <li>Weighted Graphs and Shortest Path Algorithms</li>
        <li>Graph Neural Networks (GNNs)</li>
    </ul>
    <p><b>Module 9: Matrix Factorization and Decomposition</b></p>
    <ul>
        <li>Principal Component Analysis (PCA)</li>
        <li>Non-Negative Matrix Factorization (NMF)</li>
        <li>LU, QR, and Cholesky Decompositions</li>
        <li>Applications of Matrix Decompositions in Machine Learning</li>
    </ul>
    <p><b>Module 10: Advanced Topics in AI/ML Math</b></p>
    <ul>
        <li>Tensor Calculus for Deep Learning</li>
        <li>Advanced Optimization Algorithms (Adam, Adagrad, RMSProp)</li>
        <li>Deep Learning Theory: Universal Approximation Theorem</li>
        <li>Understanding Neural Network Complexity and Overfitting</li>
    </ul>
    <p><b>Module 11: Capstone Project and Review</b></p>
    <ul>
        <li>AI/ML Mathematics Capstone Project</li>
        <li>Application of Mathematical Concepts to Real-World ML Models</li>
        <li>Final Review and Q&A</li>
    </ul>""")

@app.route('/courses/cybersecurity')
def cybersecurity_course():
    return html_template.format(title="Cybersecurity", bg_color="#F4CCCC", content="""
    <p><b>Module 1: Introduction to Cybersecurity</b></p>
    <ul>
        <li>Importance of Cybersecurity</li>
        <li>Cybersecurity Landscape and Threats</li>
        <li>Key Concepts: Confidentiality, Integrity, Availability (CIA)</li>
        <li>Cybersecurity Domains and Career Paths</li>
    </ul>
    <p><b>Module 2: Networking Fundamentals</b></p>
    <ul>
        <li>Basics of Networking: Protocols, OSI Model, TCP/IP</li>
        <li>IP Addressing and Subnetting</li>
        <li>DNS, DHCP, and NAT</li>
        <li>Network Devices and Topologies</li>
        <li>Introduction to Firewalls and VPNs</li>
    </ul>
    <p><b>Module 3: Cyber Threats and Vulnerabilities</b></p>
    <ul>
        <li>Types of Cyber Threats: Malware, Phishing, Ransomware</li>
        <li>Social Engineering Attacks</li>
        <li>Vulnerabilities and Exploits</li>
        <li>Threat Intelligence and Incident Response</li>
    </ul>
    <p><b>Module 4: Security Architecture and Design</b></p>
    <ul>
        <li>Principles of Secure System Design</li>
        <li>Security Controls: Preventive, Detective, Corrective</li>
        <li>Access Control Models: DAC, MAC, RBAC</li>
        <li>Secure Software Development Lifecycle (SSDLC)</li>
        <li>Cryptography Basics: Encryption, Decryption, and Hashing</li>
    </ul>
    <p><b>Module 5: Cryptography and PKI</b></p>
    <ul>
        <li>Symmetric and Asymmetric Encryption</li>
        <li>Public Key Infrastructure (PKI)</li>
        <li>Digital Signatures and Certificates</li>
        <li>SSL/TLS Protocols</li>
        <li>Hashing Algorithms: MD5, SHA</li>
    </ul>
    <p><b>Module 6: Operating System and Application Security</b></p>
    <ul>
        <li>Secure Configuration of Operating Systems (Windows/Linux)</li>
        <li>Patch Management and Updates</li>
        <li>Secure Coding Practices</li>
        <li>Web Application Security: OWASP Top 10</li>
        <li>Database Security</li>
    </ul>
    <p><b>Module 7: Network Security</b></p>
    <ul>
        <li>Intrusion Detection and Prevention Systems (IDS/IPS)</li>
        <li>Network Monitoring and Traffic Analysis</li>
        <li>Securing Wireless Networks</li>
        <li>Firewalls and VPN Configurations</li>
        <li>Zero Trust Architecture</li>
    </ul>
    <p><b>Module 8: Ethical Hacking and Penetration Testing</b></p>
    <ul>
        <li>Ethical Hacking Overview and Rules of Engagement</li>
        <li>Reconnaissance and Footprinting</li>
        <li>Scanning and Enumeration</li>
        <li>Exploitation and Post-Exploitation Techniques</li>
        <li>Vulnerability Assessment Tools (e.g., Nmap, Nessus)</li>
    </ul>
    <p><b>Module 9: Security Operations and Incident Response</b></p>
    <ul>
        <li>Security Operations Center (SOC) Roles</li>
        <li>Incident Detection and Response Process</li>
        <li>Forensic Investigation Basics</li>
        <li>Log Analysis and SIEM Tools (e.g., Splunk)</li>
        <li>Threat Hunting Techniques</li>
    </ul>
    <p><b>Module 10: Cloud and IoT Security</b></p>
    <ul>
        <li>Cloud Security Fundamentals</li>
        <li>Shared Responsibility Model</li>
        <li>Securing AWS, Azure, and GCP Environments</li>
        <li>IoT Security Challenges</li>
        <li>Identity and Access Management (IAM) in the Cloud</li>
    </ul>
    <p><b>Module 11: Governance, Risk, and Compliance (GRC)</b></p>
    <ul>
        <li>Introduction to GRC Frameworks</li>
        <li>Risk Assessment and Management</li>
        <li>Security Policies and Standards (ISO 27001, NIST, GDPR)</li>
        <li>Audit and Compliance Requirements</li>
        <li>Business Continuity and Disaster Recovery</li>
    </ul>
    <p><b>Module 12: Cybersecurity Tools and Techniques</b></p>
    <ul>
        <li>Security Tools: Wireshark, Metasploit, Burp Suite</li>
        <li>Scripting for Security Automation (Python/Bash)</li>
        <li>Password Cracking and Mitigation</li>
        <li>Malware Analysis Basics</li>
        <li>Endpoint Detection and Response (EDR)</li>
    </ul>
    <p><b>Module 13: Emerging Trends in Cybersecurity</b></p>
    <ul>
        <li>Artificial Intelligence and Machine Learning in Security</li>
        <li>Quantum Cryptography</li>
        <li>Cybersecurity for Blockchain</li>
        <li>Ransomware Trends and Defense Strategies</li>
        <li>Security Challenges in 5G Networks</li>
    </ul>
    <p><b>Module 14: Capstone Project and Review</b></p>
    <ul>
        <li>Real-World Cybersecurity Project</li>
        <li>Risk Assessment and Threat Modeling</li>
        <li>Implementation of Security Controls</li>
        <li>Incident Response Simulation</li>
        <li>Presentation and Feedback</li>
    </ul>
    """)

@app.route('/courses/quantum-computing')
def quantum_computing_course():
    return html_template.format(title="Quantum Computing", bg_color="#D9EAD3", content="""
    <p><b>Module 1: Introduction to Quantum Computing</b></p>
    <ul>
        <li>History and Evolution of Quantum Computing</li>
        <li>Classical vs. Quantum Computing</li>
        <li>Key Concepts: Superposition, Entanglement, Interference</li>
        <li>Applications of Quantum Computing</li>
    </ul>
    <p><b>Module 2: Mathematics for Quantum Computing</b></p>
    <ul>
        <li>Linear Algebra Essentials</li>
        <li>Probability Theory Basics</li>
        <li>Dirac Notation and Quantum States</li>
        <li>Unitary Operators and Quantum Gates</li>
    </ul>
    <p><b>Module 3: Quantum Mechanics Fundamentals</b></p>
    <ul>
        <li>Postulates of Quantum Mechanics</li>
        <li>The Schrödinger Equation</li>
        <li>Quantum Measurement and Observables</li>
        <li>Quantum State Collapse</li>
    </ul>
    <p><b>Module 4: Quantum Gates and Circuits</b></p>
    <ul>
        <li>Single-Qubit Gates: X, Y, Z, H, S, T</li>
        <li>Multi-Qubit Gates: CNOT, SWAP, Toffoli</li>
        <li>Quantum Circuit Design and Simulation</li>
        <li>Quantum Algorithms and Circuit Optimization</li>
    </ul>
    <p><b>Module 5: Quantum Algorithms</b></p>
    <ul>
        <li>Deutsch-Jozsa Algorithm</li>
        <li>Grover’s Search Algorithm</li>
        <li>Shor’s Algorithm for Factoring</li>
        <li>Quantum Phase Estimation</li>
        <li>Quantum Fourier Transform (QFT)</li>
    </ul>
    <p><b>Module 6: Quantum Computing Hardware</b></p>
    <ul>
        <li>Quantum Bits (Qubits) and Physical Realizations</li>
        <li>Quantum Computing Architectures (Superconducting, Trapped Ions, etc.)</li>
        <li>Quantum Error Correction and Fault-Tolerant Computing</li>
        <li>Noise and Decoherence in Quantum Systems</li>
    </ul>
    <p><b>Module 7: Quantum Machine Learning</b></p>
    <ul>
        <li>Quantum Data Encoding</li>
        <li>Variational Quantum Circuits for ML</li>
        <li>Quantum Neural Networks</li>
        <li>Hybrid Quantum-Classical Machine Learning</li>
    </ul>
    <p><b>Module 8: Quantum Cryptography</b></p>
    <ul>
        <li>Quantum Key Distribution (QKD)</li>
        <li>BB84 and E91 Protocols</li>
        <li>Post-Quantum Cryptography</li>
        <li>Implications for Modern Cryptographic Systems</li>
    </ul>
    <p><b>Module 9: Programming for Quantum Computing</b></p>
    <ul>
        <li>Introduction to Qiskit (IBM Quantum)</li>
        <li>Creating Quantum Circuits in Python</li>
        <li>Running Simulations on Quantum Devices</li>
        <li>Exploring Other Quantum Frameworks (Cirq, PyQuil, PennyLane)</li>
    </ul>
    <p><b>Module 10: Advanced Quantum Topics</b></p>
    <ul>
        <li>Adiabatic Quantum Computing</li>
        <li>Quantum Annealing (D-Wave Systems)</li>
        <li>Topological Quantum Computing</li>
        <li>Quantum Supremacy and Benchmarking</li>
    </ul>
    <p><b>Module 11: Applications of Quantum Computing</b></p>
    <ul>
        <li>Optimization Problems</li>
        <li>Drug Discovery and Material Science</li>
        <li>Quantum Finance</li>
        <li>Cryptanalysis and Security</li>
        <li>Quantum Internet</li>
    </ul>
    <p><b>Module 12: Challenges and Future Trends</b></p>
    <ul>
        <li>Scalability and Hardware Limitations</li>
        <li>Quantum Computing in Industry</li>
        <li>Ethical Implications of Quantum Technology</li>
        <li>Emerging Trends: Quantum AI, Quantum Sensors</li>
    </ul>
    <p><b>Module 13: Capstone Project and Review</b></p>
    <ul>
        <li>Design and Implementation of a Quantum Algorithm</li>
        <li>Simulation and Testing on Quantum Hardware</li>
        <li>Analysis of Results and Optimization</li>
        <li>Presentation and Feedback</li>
    </ul>
    """)

@app.route('/courses/r-programming')
def r_programming_course():
    return html_template.format(title="R Programming", bg_color="#D9EAD3", content="""
    <p><b>Module 1: Introduction to R Programming</b></p>
    <ul>
        <li>Introduction to R and RStudio</li>
        <li>Setting Up the R Environment</li>
        <li>Basics of R Syntax</li>
        <li>Working with R Scripts</li>
    </ul>
    <p><b>Module 2: Data Types and Structures</b></p>
    <ul>
        <li>Scalars, Vectors, and Lists</li>
        <li>Matrices and Arrays</li>
        <li>Data Frames and Tibbles</li>
        <li>Factors and Dates</li>
    </ul>
    <p><b>Module 3: Data Input and Output</b></p>
    <ul>
        <li>Reading Data: CSV, Excel, JSON, and Databases</li>
        <li>Writing Data to Files</li>
        <li>Web Scraping with R</li>
        <li>Using APIs for Data Retrieval</li>
    </ul>
    <p><b>Module 4: Data Manipulation</b></p>
    <ul>
        <li>Introduction to the dplyr Package</li>
        <li>Filtering, Selecting, and Mutating Data</li>
        <li>Grouping and Summarizing Data</li>
        <li>Data Merging and Joins</li>
    </ul>
    <p><b>Module 5: Data Visualization</b></p>
    <ul>
        <li>Introduction to the ggplot2 Package</li>
        <li>Creating Basic Plots: Scatter, Bar, Line, and Histograms</li>
        <li>Customizing Plots: Titles, Themes, and Annotations</li>
        <li>Advanced Visualization: Facets, Geoms, and Interactive Plots</li>
        <li>Visualization Tools: plotly, shiny, and leaflet</li>
    </ul>
    <p><b>Module 6: Statistical Analysis in R</b></p>
    <ul>
        <li>Descriptive Statistics</li>
        <li>Hypothesis Testing: t-test, ANOVA, and Chi-Square Test</li>
        <li>Correlation and Regression Analysis</li>
        <li>Time Series Analysis</li>
    </ul>
    <p><b>Module 7: R for Machine Learning</b></p>
    <ul>
        <li>Data Preprocessing and Feature Engineering</li>
        <li>Supervised Learning: Linear and Logistic Regression</li>
        <li>Classification Models: Decision Trees, Random Forests</li>
        <li>Unsupervised Learning: Clustering with K-Means, Hierarchical Clustering</li>
        <li>Model Evaluation Metrics</li>
    </ul>
    <p><b>Module 8: Working with Strings and Dates</b></p>
    <ul>
        <li>String Manipulation using stringr</li>
        <li>Regular Expressions in R</li>
        <li>Working with Dates and Times using lubridate</li>
    </ul>
    <p><b>Module 9: R for Big Data and Databases</b></p>
    <ul>
        <li>Introduction to Big Data with data.table</li>
        <li>Connecting R with SQL Databases</li>
        <li>Handling Large Datasets in R</li>
        <li>Integration with Hadoop and Spark</li>
    </ul>
    <p><b>Module 10: R for Advanced Analytics</b></p>
    <ul>
        <li>Text Mining with R</li>
        <li>Sentiment Analysis</li>
        <li>Network Analysis</li>
        <li>Survival Analysis</li>
    </ul>
    <p><b>Module 11: R for Reporting and Automation</b></p>
    <ul>
        <li>Creating Reports with R Markdown</li>
        <li>Automating Tasks with R Scripts</li>
        <li>Building Dashboards with Shiny</li>
        <li>Exporting Reports to PDF, HTML, and Word</li>
    </ul>
    <p><b>Module 12: R Packages and Ecosystem</b></p>
    <ul>
        <li>Installing and Managing Packages</li>
        <li>Overview of Popular R Packages: tidyverse, caret, mlr</li>
        <li>Writing Your Own R Packages</li>
        <li>Contributing to the R Community</li>
    </ul>
    <p><b>Module 13: Advanced Topics in R</b></p>
    <ul>
        <li>Functional Programming in R</li>
        <li>Object-Oriented Programming with R (S3, S4, R6)</li>
        <li>Parallel Computing in R</li>
        <li>Debugging and Performance Optimization</li>
    </ul>
    <p><b>Module 14: Capstone Project and Review</b></p>
    <ul>
        <li>End-to-End Data Analysis Project</li>
        <li>Data Cleaning, Visualization, and Analysis</li>
        <li>Machine Learning Model Development</li>
        <li>Report Creation and Presentation</li>
    </ul>
    """)

@app.route('/courses/bi')
def business_intelligence_course():
    return html_template.format(title="Business Intelligence (BI)", bg_color="#E0F7FA", content="""
    <p><b>Module 1: Introduction to Business Intelligence (BI)</b></p>
    <ul>
        <li>Overview of Business Intelligence</li>
        <li>Importance of BI in Decision-Making</li>
        <li>Components of BI Systems</li>
        <li>BI vs. Business Analytics</li>
    </ul>
    <p><b>Module 2: Data Warehousing Fundamentals</b></p>
    <ul>
        <li>Introduction to Data Warehousing</li>
        <li>ETL Process: Extract, Transform, Load</li>
        <li>Data Warehouse Architectures (Star and Snowflake Schemas)</li>
        <li>Online Analytical Processing (OLAP)</li>
    </ul>
    <p><b>Module 3: Data Visualization and Reporting</b></p>
    <ul>
        <li>Importance of Data Visualization</li>
        <li>Principles of Effective Dashboards</li>
        <li>Introduction to BI Tools (e.g., Power BI, Tableau)</li>
        <li>Creating Reports and Dashboards</li>
    </ul>
    <p><b>Module 4: BI Tools and Platforms</b></p>
    <ul>
        <li>Overview of BI Tools: Power BI, Tableau, QlikView</li>
        <li>Introduction to Microsoft Power BI</li>
        <li>Introduction to Tableau</li>
    </ul>
    <p><b>Module 5: Data Integration and ETL Tools</b></p>
    <ul>
        <li>Introduction to ETL Tools (e.g., Talend, Informatica)</li>
        <li>Data Cleaning and Preparation</li>
        <li>Data Integration from Multiple Sources</li>
        <li>Automating ETL Workflows</li>
    </ul>
    <p><b>Module 6: Data Modeling for BI</b></p>
    <ul>
        <li>Basics of Data Modeling</li>
        <li>Fact and Dimension Tables</li>
        <li>Designing Star and Snowflake Schemas</li>
        <li>Data Modeling in Power BI</li>
    </ul>
    <p><b>Module 7: Advanced BI Concepts</b></p>
    <ul>
        <li>Data Blending and Joins</li>
        <li>Calculated Fields and Measures</li>
        <li>Row-Level Security in BI Tools</li>
        <li>Custom Visualizations and Advanced Formatting</li>
    </ul>
    <p><b>Module 8: Predictive Analytics in BI</b></p>
    <ul>
        <li>Overview of Predictive Analytics</li>
        <li>Incorporating Machine Learning Models in BI Tools</li>
        <li>Time Series Forecasting</li>
        <li>Sentiment Analysis in BI</li>
    </ul>
    <p><b>Module 9: Real-Time BI and Big Data Integration</b></p>
    <ul>
        <li>Real-Time BI and Streaming Analytics</li>
        <li>Integration with Big Data Technologies (e.g., Hadoop, Spark)</li>
        <li>Working with Cloud BI Platforms (e.g., AWS QuickSight, Google Data Studio)</li>
        <li>Handling Large Datasets in BI Tools</li>
    </ul>
    <p><b>Module 10: BI Strategy and Governance</b></p>
    <ul>
        <li>Developing a BI Strategy for Organizations</li>
        <li>BI Governance and Compliance</li>
        <li>Data Privacy and Security in BI</li>
        <li>Monitoring and Evaluating BI Performance</li>
    </ul>
    <p><b>Module 11: Emerging Trends in BI</b></p>
    <ul>
        <li>Self-Service BI</li>
        <li>AI-Driven BI and Natural Language Processing</li>
        <li>Mobile BI Solutions</li>
        <li>Embedded BI in Applications</li>
    </ul>
    <p><b>Module 12: Capstone Project and Review</b></p>
    <ul>
        <li>Developing an End-to-End BI Solution</li>
        <li>Feedback and Best Practices</li>
    </ul>
    """)

@app.route('/courses/fintech')
def fintech_course():
    return html_template.format(title="Fintech", bg_color="#F4CCCC", content="""
    <p><b>Module 1: Introduction to Fintech</b></p>
    <ul>
        <li>Overview of Fintech and Its Importance</li>
        <li>Evolution of Financial Technology</li>
        <li>Key Areas of Fintech: Payments, Lending, Wealth Management, and Insurtech</li>
        <li>Fintech Ecosystem and Stakeholders</li>
    </ul>
    <p><b>Module 2: Financial Systems and Banking</b></p>
    <ul>
        <li>Traditional vs. Modern Financial Systems</li>
        <li>Banking Operations and Services</li>
        <li>Payment Systems and Infrastructure (SWIFT, ACH, and RTGS)</li>
        <li>Core Banking Systems and APIs</li>
    </ul>
    <p><b>Module 3: Blockchain and Cryptocurrencies</b></p>
    <ul>
        <li>Basics of Blockchain Technology</li>
        <li>Cryptocurrency Fundamentals: Bitcoin, Ethereum, and Others</li>
        <li>Decentralized Finance (DeFi) Overview</li>
        <li>Smart Contracts and Their Applications</li>
    </ul>
    <p><b>Module 4: Payment Technologies</b></p>
    <ul>
        <li>Evolution of Digital Payments</li>
        <li>Payment Gateways and Processors</li>
        <li>Mobile Payments and Wallets (e.g., Apple Pay, Google Pay)</li>
        <li>Cross-Border Payments and Challenges</li>
    </ul>
    <p><b>Module 5: Data Analytics in Fintech</b></p>
    <ul>
        <li>Role of Data Analytics in Financial Services</li>
        <li>Predictive Analytics in Credit Scoring and Risk Assessment</li>
        <li>Fraud Detection using Data Analytics</li>
        <li>Customer Segmentation and Personalization</li>
    </ul>
    <p><b>Module 6: Artificial Intelligence and Machine Learning in Fintech</b></p>
    <ul>
        <li>Applications of AI/ML in Financial Services</li>
        <li>Chatbots for Customer Service in Fintech</li>
        <li>Algorithmic Trading and Robo-Advisors</li>
        <li>Credit Risk Modeling using ML</li>
    </ul>
    <p><b>Module 7: RegTech and Compliance</b></p>
    <ul>
        <li>Overview of Regulatory Technology (RegTech)</li>
        <li>Anti-Money Laundering (AML) and Know Your Customer (KYC) Processes</li>
        <li>Data Privacy Laws and Compliance (GDPR, CCPA)</li>
        <li>Role of Technology in Risk and Compliance Management</li>
    </ul>
    <p><b>Module 8: Peer-to-Peer (P2P) and Crowdfunding Platforms</b></p>
    <ul>
        <li>P2P Lending Platforms: Overview and Operations</li>
        <li>Crowdfunding Models: Reward-Based, Equity-Based, and Donation-Based</li>
        <li>Challenges and Regulatory Frameworks for P2P and Crowdfunding</li>
    </ul>
    <p><b>Module 9: Insurtech and WealthTech</b></p>
    <ul>
        <li>Introduction to Insurtech: Innovations in Insurance Technology</li>
        <li>Applications of AI/ML in Insurance: Claims Processing and Risk Assessment</li>
        <li>Overview of WealthTech: Robo-Advisors and Portfolio Management</li>
        <li>Trends in Wealth Management Technology</li>
    </ul>
    <p><b>Module 10: Cybersecurity in Fintech</b></p>
    <ul>
        <li>Importance of Cybersecurity in Financial Services</li>
        <li>Common Cyber Threats: Phishing, Ransomware, and Fraud</li>
        <li>Encryption and Secure Transaction Protocols</li>
        <li>Risk Mitigation Strategies for Fintech Platforms</li>
    </ul>
    <p><b>Module 11: Open Banking and APIs</b></p>
    <ul>
        <li>Introduction to Open Banking</li>
        <li>Role of APIs in Fintech Integration</li>
        <li>Case Studies: Successful Open Banking Implementations</li>
        <li>Challenges and Risks of Open Banking</li>
    </ul>
    <p><b>Module 12: Fintech Startups and Innovation</b></p>
    <ul>
        <li>Building a Fintech Startup: Challenges and Opportunities</li>
        <li>Trends in Fintech Innovation: BNPL (Buy Now, Pay Later), Embedded Finance</li>
        <li>Case Studies: Successful Fintech Companies</li>
        <li>Importance of UX/UI in Fintech Applications</li>
    </ul>
    <p><b>Module 13: Emerging Technologies in Fintech</b></p>
    <ul>
        <li>Quantum Computing in Finance</li>
        <li>Role of IoT in Financial Services</li>
        <li>Edge Computing for Real-Time Financial Decisions</li>
        <li>Future of Biometric Payments</li>
    </ul>
    <p><b>Module 14: Capstone Project and Review</b></p>
    <ul>
        <li>Designing and Implementing a Fintech Solution</li>
        <li>Project Presentation and Feedback</li>
    </ul>
    """)

@app.route('/courses/blockchain')
def blockchain_course():
    return html_template.format(title="Blockchain", bg_color="#D9EAD3", content="""
    <p><b>Module 1: Introduction to Blockchain</b></p>
    <ul>
        <li>Basics of Blockchain Technology</li>
        <li>Key Concepts: Distributed Ledger, Consensus, and Decentralization</li>
        <li>History and Evolution of Blockchain</li>
        <li>Types of Blockchain: Public, Private, and Consortium</li>
    </ul>
    <p><b>Module 2: Blockchain Architecture and Components</b></p>
    <ul>
        <li>Structure of a Blockchain: Blocks, Transactions, and Chains</li>
        <li>Cryptographic Hashing and Its Role in Blockchain</li>
        <li>Merkle Trees and Data Integrity</li>
        <li>Peer-to-Peer Networks and Nodes</li>
    </ul>
    <p><b>Module 3: Consensus Mechanisms</b></p>
    <ul>
        <li>Introduction to Consensus Mechanisms</li>
        <li>Proof of Work (PoW)</li>
        <li>Proof of Stake (PoS) and Variants</li>
        <li>Delegated Proof of Stake (DPoS) and Practical Byzantine Fault Tolerance (PBFT)</li>
    </ul>
    <p><b>Module 4: Bitcoin and Cryptocurrency Fundamentals</b></p>
    <ul>
        <li>Overview of Bitcoin and Its Ecosystem</li>
        <li>Mining: How Bitcoin Transactions Are Validated</li>
        <li>Bitcoin Wallets and Keys</li>
        <li>Security Features in Bitcoin</li>
    </ul>
    <p><b>Module 5: Ethereum and Smart Contracts</b></p>
    <ul>
        <li>Introduction to Ethereum and Its Differences from Bitcoin</li>
        <li>Smart Contracts: Definition, Use Cases, and Limitations</li>
        <li>Ethereum Virtual Machine (EVM)</li>
        <li>ERC Standards: ERC-20, ERC-721, and ERC-1155</li>
    </ul>
    <p><b>Module 6: Decentralized Applications (DApps)</b></p>
    <ul>
        <li>What Are DApps and Their Benefits?</li>
        <li>Tools and Frameworks for DApp Development (e.g., Truffle, Hardhat)</li>
        <li>Building a Simple DApp</li>
        <li>Challenges in DApp Adoption</li>
    </ul>
    <p><b>Module 7: Blockchain Development</b></p>
    <ul>
        <li>Introduction to Solidity Programming</li>
        <li>Writing and Deploying Smart Contracts</li>
        <li>Debugging and Testing Smart Contracts</li>
        <li>Gas Fees and Optimization Techniques</li>
    </ul>
    <p><b>Module 8: Blockchain Security</b></p>
    <ul>
        <li>Common Vulnerabilities in Smart Contracts</li>
        <li>Tools for Auditing Smart Contracts (e.g., MythX, Slither)</li>
        <li>Cryptographic Principles: Digital Signatures, PKI, and Encryption</li>
        <li>Security Best Practices in Blockchain Development</li>
    </ul>
    <p><b>Module 9: Blockchain Platforms and Frameworks</b></p>
    <ul>
        <li>Overview of Popular Platforms: Hyperledger, Corda, and Polkadot</li>
        <li>Comparison of Blockchain Platforms</li>
        <li>Setting Up a Private Blockchain Using Hyperledger Fabric</li>
        <li>Introduction to Layer 2 Solutions and Sidechains</li>
    </ul>
    <p><b>Module 10: Decentralized Finance (DeFi)</b></p>
    <ul>
        <li>What Is DeFi and Why It Matters?</li>
        <li>Key DeFi Use Cases: Lending, Borrowing, and Staking</li>
        <li>Role of Stablecoins in DeFi</li>
        <li>Risks and Challenges in DeFi Applications</li>
    </ul>
    <p><b>Module 11: Blockchain in Enterprises</b></p>
    <ul>
        <li>Use Cases of Blockchain in Supply Chain, Healthcare, and Finance</li>
        <li>Tokenization of Assets</li>
        <li>Blockchain as a Service (BaaS) Providers</li>
        <li>Integration of Blockchain with IoT and AI</li>
    </ul>
    <p><b>Module 12: Non-Fungible Tokens (NFTs)</b></p>
    <ul>
        <li>Introduction to NFTs and Their Characteristics</li>
        <li>Creating and Minting NFTs</li>
        <li>NFT Marketplaces (e.g., OpenSea, Rarible)</li>
        <li>Applications of NFTs Beyond Art and Collectibles</li>
    </ul>
    <p><b>Module 13: Emerging Trends in Blockchain</b></p>
    <ul>
        <li>Introduction to Web3 and the Metaverse</li>
        <li>Cross-Chain Interoperability and Bridges</li>
        <li>Blockchain Scalability Solutions (Sharding, Rollups)</li>
        <li>Quantum Computing and Its Impact on Blockchain</li>
    </ul>
    <p><b>Module 14: Capstone Project and Review</b></p>
    <ul>
        <li>Developing a Blockchain-Based Solution</li>
        <li>Project Presentation and Feedback</li>
    </ul>
    """)

@app.route('/courses/c-programming')
def c_programming_course():
    return html_template.format(title="C Programming", bg_color="#D9EAD3", content="""
    <p><b>Module 1: Introduction to C Programming</b></p>
    <ul>
        <li>History and Evolution of C</li>
        <li>Basic Structure of a C Program</li>
        <li>Compiling and Running a C Program</li>
        <li>Understanding Header Files and Libraries</li>
    </ul>
    <p><b>Module 2: Data Types and Operators</b></p>
    <ul>
        <li>Variables and Constants</li>
        <li>Primitive Data Types (int, float, char, etc.)</li>
        <li>Operators: Arithmetic, Relational, Logical, Bitwise</li>
        <li>Type Conversion and Type Casting</li>
    </ul>
    <p><b>Module 3: Control Flow Statements</b></p>
    <ul>
        <li>Conditional Statements (if, if-else, switch)</li>
        <li>Looping Statements (for, while, do-while)</li>
        <li>Jump Statements (break, continue, goto)</li>
    </ul>
    <p><b>Module 4: Functions in C</b></p>
    <ul>
        <li>Defining and Calling Functions</li>
        <li>Function Parameters and Return Types</li>
        <li>Recursion in C</li>
        <li>Storage Classes (auto, static, extern, register)</li>
    </ul>
    <p><b>Module 5: Arrays and Strings</b></p>
    <ul>
        <li>One-Dimensional and Multi-Dimensional Arrays</li>
        <li>String Handling Functions</li>
        <li>Pointers and Strings</li>
        <li>Passing Arrays to Functions</li>
    </ul>
    <p><b>Module 6: Pointers in C</b></p>
    <ul>
        <li>Introduction to Pointers</li>
        <li>Pointer Arithmetic</li>
        <li>Pointers and Arrays</li>
        <li>Dynamic Memory Allocation (malloc, calloc, free)</li>
    </ul>
    <p><b>Module 7: Structures and Unions</b></p>
    <ul>
        <li>Defining and Using Structures</li>
        <li>Array of Structures</li>
        <li>Nested Structures</li>
        <li>Difference Between Structures and Unions</li>
    </ul>
    <p><b>Module 8: File Handling in C</b></p>
    <ul>
        <li>File Operations: Read and Write</li>
        <li>Working with Text and Binary Files</li>
        <li>File Pointers and Functions (fopen, fclose, fread, fwrite)</li>
        <li>Command Line Arguments</li>
    </ul>
    <p><b>Module 9: Advanced Topics in C</b></p>
    <ul>
        <li>Bitwise Operations and Their Applications</li>
        <li>Preprocessors and Macros</li>
        <li>Linked Lists: Singly, Doubly, Circular</li>
        <li>Memory Management Techniques</li>
    </ul>
    <p><b>Module 10: Capstone Project</b></p>
    <ul>
        <li>Developing a C-Based Application</li>
        <li>Project Presentation and Code Review</li>
    </ul>
    """)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
