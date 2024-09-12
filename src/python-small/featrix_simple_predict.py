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
import json
import socket
import uuid
import warnings
from datetime import datetime
from typing import Optional

import requests

class FeatrixConnectionError(Exception):
    def __init__(self, url, message):
        # logger.error("Connection error for url %s: __%s__" % (url, message))
        super().__init__("Connection error for URL %s: __%s__" % (url, message))

class FeatrixBadApiKeyError(Exception):
    def __init__(self, msg):
        self.message = msg

class Featrix:
    def __init__(
        self,
        url: str = "https://app.featrix.com",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        allow_unencrypted_http: bool = False,
        debug = False
    ):
        self.debug = debug
        self.client_id = client_id
        self.client_secret = client_secret
        self.url = self._validate_url(url, allow_unencrypted_http)
        self.hostname = socket.gethostname()
        self._generate_bearer_token()
        if self._current_bearer_token is None:
            raise Exception(
                "Your client id and client secret pair are invalid."
            )

    def _generate_bearer_token(self):
        if self.debug:
            print("_generate_bearer_token entered")
        headers = self._featrix_headers(bearer_generate=True)

        payload = json.dumps(
                {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                })

        response = requests.post(
            f"{self.url}/mosaic/keyauth/jwt",
            headers=headers,
            data=payload,
        )

        if response.status_code == 200:
            # The response will have your JWT in it -- you can use that now for 24 hours
            # If this was a guest access, it will also have the client-id/client-secret for reuse
            body = response.json()
            self._current_bearer_token = body["jwt"]
            self._current_bearer_token_expiration = datetime.fromisoformat(
                body["expiration"]
            )
            if self.debug:
                print("bearer looks good")
        else:
            raise FeatrixBadApiKeyError(
                f"Failed to create authorization token from API Key: {response.status_code}. "
                "Are your client id and client secret correct?"
            )


    def _featrix_headers(
        self,
        extra: Optional[dict] = None,
        json_request: Optional[bool] = False,
        bearer_generate: Optional[bool] = False,
        **kwargs,
    ):
        # FIXME: do we use this anymore?
        if json_request:
            headers = {
                "Content-type": "application/json",
                "Accept": "text/plain",
                "X-request-id": str(uuid.uuid1()),
                "X-hostname": self.hostname,
            }
        else:
            headers = {}

        if extra is not None:
            headers.update(extra)
        headers.update(kwargs)
        if not bearer_generate:
            if self._current_bearer_token is None or (
                self._current_bearer_token_expiration
                and self._current_bearer_token_expiration < datetime.now()
            ):  
                self._generate_bearer_token()
            if self._current_bearer_token is None:
                raise FeatrixBadApiKeyError(
                    "You ApiKey seems to have been invalidated, please create another one"
                )
            # Don't let old/handed-in headers overwrite the auth token
            headers["Authorization"] = f"Bearer {self._current_bearer_token}"
        return headers
    def _validate_url(self, url: str, allow_unencrypted_http) -> str:
        if url is None:
            raise RuntimeError(
                f"url argument to {self.__class__.__name__} must be in "
                "the form of https://host:port"
            )
        url = url.rstrip("/")  # Remove trailing slashes
        self._validate_url_scheme(url, allow_unencrypted_http)
        return f"{url}/api"

    def _validate_url_scheme(self, url: str, allow_unencrypted_http):
        if not url.startswith("http://") and not url.startswith("https://"):  # noqa
            raise RuntimeError(
                "url argument to must be in the form of https://host:port"
            )
        elif url.startswith("http://"):  # noqa - dev mode
            self._validate_localhost_url(url, allow_unencrypted_http)
        return

    def _validate_localhost_url(self, url: str, allow_unencrypted_http):
        if not allow_unencrypted_http:
            localhost_prefixes = ["localhost", "127.0.0.1", "::1"]
            for lh in localhost_prefixes:
                lh = "http://" + lh  # noqa dev mode
                # Capture with both a specified port or a slash.
                if url == lh or url.startswith(lh + "/") or url.startswith(lh + ":"):
                    break
            else:
                raise RuntimeError(
                    "Non-HTTPS only supported for localhost without setting `allow_unencrypted_http`"
                )
        return

    def predict(self, neural_function_id, query):
        assert isinstance(query, dict) or isinstance(query, list)
        if isinstance(query, dict):
            query = [query]
        retries = 3
        url  = f"{self.url}/neural/models/prediction"
        headers = self._featrix_headers()
        if self.debug:
            print("HTTP: ", url)
            print("headers: ", headers)
        try:
            payload = { "job_type": "model-prediction",
                        "model_id": neural_function_id,
                        "query": query  
                      }
            response = requests.request(
                "POST", url, headers=headers, json=payload
            )
            if self.debug:
                print(f"Response status: {response.status_code}")
            # print(f"response back {response}")
        except requests.exceptions.Timeout:
            if self.debug:
                print("Response exception: Timeout")
            retries -= 1
            warnings.warn(f"Request timed out, retrying (will retry {retries} times")
            response = None
        except requests.exceptions.HTTPError as err:
            if self.debug:
                print(f"Response exception: {err}")
            raise FeatrixConnectionError(url, f"http error: {err}")
        except requests.RequestException as e:
            if self.debug:
                print(f"Response exception: {e}")
            raise FeatrixConnectionError(url, f"request error: {e}")
        except Exception as e:
            if self.debug:
                print(f"Response exception: {e}")
            raise FeatrixConnectionError(url, f"Unknown error {e}")
        return response.json()
