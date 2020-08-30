from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      hs = request.form["house_size"]
      nor = request.form["num_of_bedrooms"]
      filename = 'model.sav'
      loaded_model = pickle.load(open(filename, 'rb'))
      X_test = np.array([[hs,nor]]).astype(np.float64)
      price = loaded_model.predict(X_test)
      price = np.round(price,2)
      return jsonify(result=price.tolist())
      
if __name__ == '__main__':
   app.run()


#Note
#if set app.run(debug = True), raise the problem --> address already in use 
#How to fix: 
# ps aux | grep python
# sudo kill -9 PID