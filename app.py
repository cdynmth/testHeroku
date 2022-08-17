from flask import Flask, render_template,request, url_for
from helper_function import run_model

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def iris_pred():

    if request.method == "POST":
        sp_length =  float(request.form['feature 1'])
        sp_width =  float(request.form['feature 2'])
        pt_length =  float(request.form['feature 3'])
        pt_width =  float(request.form['feature 4'])

        list_features = [sp_length,sp_width, pt_length,pt_width]
        name=run_model(list_features)
        return render_template('main.html',pred=name)
    else:
        return render_template('main.html')

@app.route('/play', methods=['POST', 'GET'])
def video():
    return render_template('video.html')


if __name__ == "__main__":
    app.run(debug=True, port=8732)