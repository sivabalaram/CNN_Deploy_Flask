from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image

UPLOAD_FOLDER = '/upload'

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/')
def dorc(test_image):
    model=tf.keras.models.load_model("CNN_Model.h5") 
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    prediction=result[0][0]
    if result[0][0]==1:
        prediction='Dog'
    else:
        prediction='Cat'
    return prediction   

@app.route('/',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        img=Image.open(request.files['image_123'].stream)
        img=img.resize((64,64))
        r=dorc(img)
        return render_template('index.html',c=r)  


if __name__ == '__main__':
    app.run(debug=True)          

