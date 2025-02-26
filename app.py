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


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
