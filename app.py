from flask import Flask

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
        <li>Python Programming</li>
        <li>Artificial Intelligence (AI)</li>
        <li>Machine Learning (ML)</li>
        <li>Deep Learning</li>
        <li>Generative AI (Gen AI)</li>
        <li>Data Science</li>
        <li>AI/ML Mathematics</li>
        <li>Cybersecurity</li>
        <li>Quantum Computing</li>
        <li>R Programming</li>
        <li>Business Intelligence (BI)</li>
        <li>Fintech</li>
        <li>Blockchain</li>
        <li>C Programming</li>
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

if __name__ == '__main__':
    app.run(debug=True)