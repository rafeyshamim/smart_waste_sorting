from flask import Flask , render_template
import flask

app = Flask(__name__)

@app.route('/')
def Home_page():
    return render_template('home.html')

@app.route('/home')
def Main_page():
    return render_template('home.html') 

@app.route('/about/<username>')
def profile(username):
    return f'<h1> heyy {username} </h1>'

if __name__ == '__main__':
    app.run(debug=True)