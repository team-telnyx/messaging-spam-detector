import flask
from flask import request

from spam_detector.predictor import MessageStatusPredictor

flask_app = flask.Flask(__name__)
predictor = MessageStatusPredictor()


@flask_app.route("/v1/spam_checks", methods=['POST'])
def check_spam():
    req_payload = request.json
    msg_body = req_payload.get("message_body", "")
    spam_resp = predictor.predict(msg_body=msg_body)
    return flask.jsonify(
        data={
            "is_spam": spam_resp.is_spam,
            "error": spam_resp.error,
            "original_message": msg_body
        },
        record_type="spam_check")


if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=8080)
