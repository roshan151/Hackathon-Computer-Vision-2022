from flask import Flask, jsonify, request

# create app
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    address = request.get_json()["address"]

    try:
        # TODO: Insert get lat/lon

        # Insert model here and predict
        price = len(address)

        # Return the price rounded to nearest dollar as a string
        result = {"price": str(round(price))}

    except Exception as e:
        print("error")
        status = 400
        result = {"error": status, "error_description": f"exception, {e}"}

    return jsonify(result)


# main
if __name__ == "__main__":
    app.run(debug=True)
