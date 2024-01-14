#import statements 
from flask import Flask
from flask import render_template
from flask import request
import mysql.connector
from datetime import datetime

app = Flask(__name__)

#connecting to database 
db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database= "shehacks"
)

#creating cursor 
cursor = db.cursor()

#get the information for a certain plant 
@app.route('/getInfo', methods=['GET'])
def get_Information():
    #get the type from user selection
    plant_type = request.args.get('Type')

    #if there is one selected query the table to get the info
    if(plant_type):
        query = "SELECT Information FROM plants WHERE Type = %s"
        #execute the query 
        cursor.execute(query, (plant_type,))
        #get the data
        data = cursor.fetchall()
        #return data 
        return str(data)
    else:
        return "Please provide a valid plant type in the URL parameter 'Type'."

#get the plants that the user has discovered 
@app.route('/getFoundP', methods=['GET'])
def get_plants():
    #query 
    query = "SELECT Type FROM plants WHERE Discover = 1"
    #execute 
    cursor.execute(query)
    #get data 
    data = cursor.fetchall()
    #return data 
    return str(data)

#reset the database values
@app.route('/resetFound', methods=['POST'])
def reset_Found():
    #query 
    query = "UPDATE plants SET Discover = FALSE"
    #execute
    cursor.execute(query)
    db.commit()
    #query 
    query = "UPDATE plants SET Date = 0000-00-00"
    #execute
    cursor.execute(query)
    db.commit()
    #return statement 
    return "Update successful"

#update the date of when the plant was discovered 
@app.route('/updateDate', methods=['POST'])
def update_Date():
    cd = datetime.now().strftime('%Y-%m-%d')
    #query 
    query = "UPDATE plants SET Date = %s WHERE Type = %s"
    if request.method == 'POST':
        type = request.form.get('Type')
    var = (cd, type)
    #execute
    cursor.execute(query, var)
    db.commit()
    #return statement 
    return("Update Successful")

#update discover to show the ones the user has found 
@app.route('/updateDiscover', methods=['POST'])
def change_Discover():
    #query 
    query = "UPDATE plants SET Discover = TRUE WHERE Type = %s"
    if request.method == 'POST':
        type = request.form.get('Type')
        val = (type,)
        #execute
        cursor.execute(query, val)
        db.commit()
    #return statement 
    return "Update successful"

