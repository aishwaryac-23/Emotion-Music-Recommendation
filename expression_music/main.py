
from camera import VideoCamera
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, Response
import gunicorn
import random
import string
import os
import cv2
from werkzeug.utils import secure_filename
import mysql.connector
import csv
import datetime
import base64
import time
import shutil
import numpy as np
import pandas as pd
import imagehash
import PIL.Image
from PIL import Image
from io import BytesIO
import io
from PIL import ImageTk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import urllib.request
import urllib.parse
from urllib.request import urlopen
import webbrowser
#from midas.model_loader import default_models, load_model
import argparse
import pyttsx3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
import sklearn


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/songs'  # Folder to store songs
ALLOWED_EXTENSIONS = {'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app.secret_key = 'abcdef'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    charset="utf8",
    database="emotion_music"
)


video_camera = VideoCamera()  # Initialize camera instance

@app.route('/')
def index():
    return render_template('index.html')

video_camera = VideoCamera()

@app.route('/capture_image')
def capture_image():
    frame, emotion = video_camera.capture_frame()  # Capture image with bounding box and detect emotion

    if frame is not None:
        # Save the processed image with bounding box
        image_filename = f"static/captured/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        cv2.imwrite(image_filename, frame)

        # Fetch songs based on detected emotion
        cursor = mydb.cursor()
        cursor.execute("SELECT song FROM song WHERE emotion = %s", (emotion,))
        songs = [row[0] for row in cursor.fetchall()]
        cursor.close()

        return jsonify({"image": "/" + image_filename, "emotion": emotion, "songs": songs})

    return jsonify({"error": "Error capturing image"}), 500


@app.route('/admin',methods=['POST','GET'])
def admin():
    
    
    msg=""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        
        if account:
            session['username'] = username
            session['user_type'] = 'admin'
            msg="success"
            return redirect(url_for('pro1'))
        else:
            msg="fail"
        

    return render_template('admin.html',msg=msg)



@app.route('/login',methods=['POST','GET'])
def login():
    
    
    msg=""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM user WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        
        if account:
            session['username'] = username
            session['user_type'] = 'user'
            msg="success"
            return redirect(url_for('add_song'))
        else:
            msg="fail"
        

    return render_template('login.html',msg=msg)


@app.route('/regsiter',methods=['POST','GET'])
def regsiter():
    msg=""
    st=""
    name=""
    email=""
    mess=""
    reg_no=""
    password=""
    if request.method=='POST':

        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        address=request.form['address']
        username=request.form['username']
        password=request.form['password']

        
        now = datetime.datetime.now()
        date_join=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        mycursor.execute("SELECT count(*) FROM user where username=%s",(username, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM user")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO user(id, name, mobile, email, address, username, password,date_join) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, mobile, email, address, username, password,date_join)
            mycursor.execute(sql, val)
            mydb.commit()

            msg="success"
            st="1"
            mess = f"Reminder: Hi {name}, your username is {reg_no} and password is {password}!"
            mycursor.close()
        else:
            msg="fail"
            
    return render_template('regsiter.html', msg=msg)


        
@app.route('/add_song', methods=['POST', 'GET'])
def add_song():

    username=session.get('username')

    msg = ""
    if request.method=='POST':
        emotion=request.form['emotion']        
        file = request.files['song']
        if file and allowed_file(file.filename):
            # Generate a random filename
            random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + ".mp3"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], random_filename)

            # Save the file
            file.save(filepath)
            now = datetime.datetime.now()
            date_join=now.strftime("%d-%m-%Y")
            mycursor = mydb.cursor()
            mycursor.execute("SELECT max(id)+1 FROM song")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO song(id, emotion, song, username, date_join) VALUES (%s, %s, %s, %s, %s)"
            val = (maxid, emotion, random_filename, username, date_join)
            mycursor.execute(sql, val)
            mydb.commit()
            msg="success"
            return redirect(url_for('songs'))

    return render_template('add_song.html', msg=msg)


@app.route('/test')
def test():
    return render_template('test.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    return jsonify({"emotion": video_camera.get_emotion()})


###############################################################################



@app.route('/pro1', methods=['GET', 'POST'])
def pro1():
    msg=""
    data=[]
    df=pd.read_csv('static/data_by_year.csv')
    dat=df.head(1100)

    for ss in dat.values:
        data.append(ss)
    
    return render_template('pro1.html',data=data)

@app.route('/pro2', methods=['GET', 'POST'])
def pro2():
    msg=""
    mem=0
    cnt=0
    cols=0
    filename = 'static/data_by_year.csv'
    data1 = pd.read_csv(filename, header=0)
    data2 = list(data1.values.flatten())
    cname=[]
    data=[]
    dtype=[]
    dtt=[]
    nv=[]
    i=0
    
    sd=len(data1)
    rows=len(data1.values)
    
    #print(data1.columns)
    col=data1.columns
    #print(data1[0])
    for ss in data1.values:
        cnt=len(ss)
        

    i=0
    while i<cnt:
        j=0
        x=0
        for rr in data1.values:
            dt=type(rr[i])
            if rr[i]!="":
                x+=1
            
            j+=1
        dtt.append(dt)
        nv.append(str(x))
        
        i+=1

    arr1=np.array(col)
    arr2=np.array(nv)
    data3=np.vstack((arr1, arr2))


    arr3=np.array(data3)
    arr4=np.array(dtt)
    
    data=np.vstack((arr3, arr4))
   
    print(data)
    cols=cnt
    mem=float(rows)*0.75

    return render_template('pro2.html',data=data, msg=msg, rows=rows, cols=cols, dtype=dtype, mem=mem)


    

@app.route('/pro3', methods=['GET', 'POST'])
def pro3():
    # Load dataset
    try:
        dataset = pd.read_csv('static/data_by_year.csv')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Feature extraction process
    num_rows = len(dataset)
    num_columns = len(dataset.columns)
    column_names = dataset.columns.tolist()
    dataset_summary = dataset.describe().to_html(classes="table table-striped")

    # Data to display
    data = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "column_names": column_names,
        "dataset_summary": dataset_summary,
    }
  
    return render_template('pro3.html',data=data)



#Classifier
class Classifier:
    def fit(self, X, y):
        n_samples, n_features = X.shape# P = X^T X
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = np.dot(X[i], X[j])
                P = cvxopt.matrix(np.outer(y, y) * K)# q = -1 (1xN)
        q = cvxopt.matrix(np.ones(n_samples) * -1)# A = y^T 
        A = cvxopt.matrix(y, (1, n_samples))# b = 0 
        b = cvxopt.matrix(0.0)# -1 (NxN)
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))# 0 (1xN)
        h = cvxopt.matrix(np.zeros(n_samples))
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)# Lagrange multipliers
        a = np.ravel(solution['x'])# Lagrange have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]# Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)# Weights
        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        
    def project(self, X):
        return np.dot(X, self.w) + self.b
    
    
    def predict(self, X):
        return np.sign(self.project(X))





# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode target if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, df.columns[:-1], le if 'le' in locals() else None



CSV_FILE = "static/data_by_year.csv"

def classify_data(csv_path):
    """Reads CSV, classifies data (dummy classifier for now), and returns features & labels."""
    df = pd.read_csv(csv_path)
    
    # Assuming first 3 columns are features, last column is label
    feature_columns = df.columns[:-1]
    label_column = df.columns[-1]

    X = df[feature_columns].values  # Features
    y = df[label_column].values  # Labels

    # Simple dummy classification (replace with ML model)
    classified_labels = np.where(y > np.median(y), "High", "Low")

    return df, classified_labels

def generate_3d_graphs(df):
    try:
        # Check for NaN or Infinite values
        if df.isna().sum().sum() > 0 or df.isin([np.inf, -np.inf]).sum().sum() > 0:
            raise ValueError("Data contains NaN or Infinite values. Please clean the data.")

        # Ensure there are enough data points for triangulation
        if len(df) < 4:
            raise ValueError("Insufficient data points for triangulation. At least 4 points required.")

        # First plot: 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c='r', marker='o')
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        ax.set_zlabel(df.columns[2])
        plt.title("3D Scatter Plot of Features")
        plt.savefig("static/graph1.png")  # Save scatter plot
        
        # Second plot: 3D surface plot using triangulation
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')

        # Try Delaunay triangulation (plot_trisurf) and catch errors
        try:
            ax2.plot_trisurf(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], cmap='viridis')
            ax2.set_xlabel(df.columns[0])
            ax2.set_ylabel(df.columns[1])
            ax2.set_zlabel(df.columns[2])
            plt.title("3D Surface Plot")
            plt.savefig("static/graph2.png")  # Save surface plot
        except RuntimeError as e:
            print(f"Error during Delaunay triangulation: {e}")
            # Provide fallback in case of error
            ax2.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c='b', marker='o')
            ax2.set_xlabel(df.columns[0])
            ax2.set_ylabel(df.columns[1])
            ax2.set_zlabel(df.columns[2])
            plt.title("3D Scatter Plot (Fallback for Triangulation Failure)")
            plt.savefig("static/graph2.png")  # Save fallback plot

    except ValueError as e:
        return f"Data Error: {e}"

@app.route('/pro4', methods=['GET', 'POST'])
def classify_and_visualize():
    """Handles CSV classification, visualization, and table display."""
    try:
        result=""

        df=""
        
        
        
        
        return render_template('pro4.html')
    
    except Exception as e:
        return str(e)  # Return any other unexpected errors



    
#####################################################################################################################


@app.route('/songs')
def songs():
    cursor = mydb.cursor(dictionary=True)  # Fetch rows as dictionaries
    cursor.execute("SELECT id, emotion, song FROM song")
    songs = cursor.fetchall()
    cursor.close()
    return render_template('songs.html', songs=songs)


@app.route('/delete_song/<int:song_id>', methods=['POST'])
def delete_song(song_id):
    cursor = mydb.cursor()

    # Retrieve the song filename before deletion
    cursor.execute("SELECT song FROM song WHERE id = %s", (song_id,))
    song = cursor.fetchone()

    if song:
        # Remove the song file from the static folder
        song_path = os.path.join(app.config['UPLOAD_FOLDER'], song[0])
        if os.path.exists(song_path):
            os.remove(song_path)

        # Delete the song entry from the database
        cursor.execute("DELETE FROM song WHERE id = %s", (song_id,))
        mydb.commit()

    cursor.close()
    return redirect(url_for('songs'))




@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True
    app.run()
