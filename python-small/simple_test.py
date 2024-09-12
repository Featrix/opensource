#  -*- coding: utf-8 -*-
#############################################################################
#
#  Copyright (c) 2024, Featrix, Inc.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#############################################################################
#
#     Welcome to...
#
#      _______ _______ _______ _______ ______ _______ ___ ___
#     |    ___|    ___|   _   |_     _|   __ \_     _|   |   |
#     |    ___|    ___|       | |   | |      <_|   |_|-     -|
#     |___|   |_______|___|___| |___| |___|__|_______|___|___|
#
#                                                 Let's embed!
#
#############################################################################
#
#  Sign up for Featrix at https://app.featrix.com/
# 
#############################################################################
#
#  Check out the docs -- you can either call the python built-in help()
#  or fire up your browser:
#
#     https://featrix-docs.readthedocs.io/en/latest/
#
#  You can also join our community Slack:
#
#     https://bits.featrix.com/slack
#
#  We'd love to hear from you: bugs, features, questions -- send them along!
#
#     hello@featrix.ai
#
#############################################################################
#
import os

from featrix_simple_predict import Featrix
from featrix_simple_predict import FeatrixBadApiKeyError
from featrix_simple_predict import FeatrixConnectionError

FEATRIX_URL = os.environ.get("FEATRIX_URL") or "https://app.featrix.com"
FEATRIX_CLIENT_ID = os.environ.get("FEATRIX_CLIENT_ID")
FEATRIX_CLIENT_SECRET = os.environ.get("FEATRIX_CLIENT_SECRET")


def main():
    ft = Featrix(
            url=FEATRIX_URL,
            client_id=FEATRIX_CLIENT_ID, 
            client_secret=FEATRIX_CLIENT_SECRET, 
            debug=True)

    query = {
        "checking_status": "no checking",
        "duration": 24,
        "credit_history": "existing paid",
        "purpose": "radio/tv",
        "credit_amount": 1376,
        "savings_status": "500<=X<1000",
        "employment": "4<=X<7",
        "installment_commitment": 4,
        "personal_status": "female div/dep/mar",
        "other_parties": "none",
        "residence_since": 1,
        "property_magnitude": "car",
        "age": 28,
        "other_payment_plans": "none",
        "housing": "own",
        "existing_credits": 1,
        "job": "skilled",
        "num_dependents": 1,
        "own_telephone": "none",
        "foreign_worker": "yes"
    }

    r = ft.predict(neural_function_id="66e2e31eb12ac3b2e7b00bca", query=query)
    print("result = ", r)
    return



if __name__ == "__main__":
    main()
