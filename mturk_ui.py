from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('mturk_ui4.html')

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, send_from_directory

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return send_from_directory('static', 'mturk_ui_pure_minimal2.html')

# if __name__ == '__main__':
#     app.run(debug=True)
