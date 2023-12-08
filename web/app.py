from flask import Flask, request
from model import prin

app=Flask(__name__)

@app.route("/",methods=['POST'])
def main():
    input = request.form["input"]
    return prin(input)
if _name_ == "_main_":
   app.run(host='127.0.0.1', port=5000)