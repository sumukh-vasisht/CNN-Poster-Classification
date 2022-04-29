from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
import os

app = Flask(__name__)

PEOPLE_FOLDER = os.path.join('static', 'test_posters')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/genre', methods = ['GET', 'POST'])
def genre():   
    if request.method=="POST":
        f = request.files['file']
        f.save('static/test_posters/test_poster.jpg')
        model_path = 'Model_4d.h5'
        model = load_model(model_path) 
        img = image.load_img('static/test_posters/test_poster.jpg',target_size=(200,150,3))
        img = image.img_to_array(img)
        img = img/255
        prob = model.predict(img.reshape(1,200,150,3))

        top_3 = np.argsort(prob[0])[:-4:-1]

        column_lookups = pd.read_csv("data/Encoded_data_column_lookup.csv", delimiter=" ")
        classes = np.asarray(column_lookups.iloc[1:29, 0])

        genres = ''

        genre0 = classes[top_3[0]]
        genre1 = classes[top_3[1]]
        genre2 = classes[top_3[2]]

        prob0 = prob[0][top_3[0]]
        prob1 = prob[0][top_3[1]]
        prob2 = prob[0][top_3[2]]

        print(genre0, genre1, genre2)

        poster_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test_poster.jpg')

        for i in range(3):
            genres += classes[top_3[i]]
            genres+='|'
            # print("{}".format(classes[top_3[i]])+" ({:.3})".format(prob[0][top_3[i]]))
        # plt.imshow(img)
        return render_template("genre.html", genre0 = genre0, genre1 = genre1, genre2 = genre2, prob0 = prob0, prob1 = prob1, prob2 = prob2, poster_path = poster_path)
    return 'Not found'
		
if __name__ == '__main__':
   app.run(debug = True)